#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <iomanip>

// --- CONFIGURACION DE TINYSTORIES-1M ---
const int VOCAB_SIZE = 50257;
const int HIDDEN_SIZE = 64;
const int NUM_LAYERS = 8;
const int NUM_HEADS = 4;
const int HEAD_DIM = HIDDEN_SIZE / NUM_HEADS;
const int END_OF_TEXT = 50256; 

// Variables globales para profiling (Edge Computing Metrics)
long long total_mac_ops_per_token = 0;
size_t max_activation_bytes = 0;

// Estructura TensorQ4 para pesos empaquetados de 4 bits
struct TensorQ4 {
    std::string name;
    uint32_t size; // Cantidad real de parametros
    float scale;
    std::vector<uint8_t> data; // Almacena 2 pesos por byte
};

struct Block {
    TensorQ4 ln_1_weight, ln_1_bias;
    TensorQ4 q_proj_w, k_proj_w, v_proj_w;
    TensorQ4 out_proj_w, out_proj_b;
    TensorQ4 ln_2_weight, ln_2_bias;
    TensorQ4 mlp_fc_w, mlp_fc_b;
    TensorQ4 mlp_proj_w, mlp_proj_b;
};

struct GPTNeoModel {
    TensorQ4 wte;
    TensorQ4 wpe;
    std::vector<Block> blocks;
    TensorQ4 ln_f_weight, ln_f_bias;
    TensorQ4 lm_head;
};

struct LayerCache {
    std::vector<std::vector<float>> k_past; 
    std::vector<std::vector<float>> v_past; 
};

struct KVCache {
    std::vector<LayerCache> layers;
    KVCache() : layers(NUM_LAYERS) {}
};

// --- OPERACIONES MATEMATICAS FUNDAMENTALES (INT4) ---

void linear_int4(const std::vector<float>& input, const TensorQ4& weight, const TensorQ4* bias, std::vector<float>& output, int in_dim, int out_dim) {
    float max_in = 0.0f;
    for (float val : input) {
        if (std::abs(val) > max_in) max_in = std::abs(val);
    }
    float in_scale = max_in > 0 ? max_in / 127.0f : 1.0f;

    for (int o = 0; o < out_dim; ++o) {
        int32_t acc = 0;
        
        for (int i = 0; i < in_dim; i += 2) {
            int data_index = (o * in_dim + i) / 2;
            uint8_t packed = weight.data[data_index];
            
            int8_t w_q0 = (packed & 0x0F);
            if (w_q0 & 0x08) w_q0 |= 0xF0;
            
            int8_t w_q1 = (packed >> 4) & 0x0F;
            if (w_q1 & 0x08) w_q1 |= 0xF0;
            
            int8_t in_q0 = static_cast<int8_t>(std::round(input[i] / in_scale)); 
            int8_t in_q1 = (i + 1 < in_dim) ? static_cast<int8_t>(std::round(input[i + 1] / in_scale)) : 0;
            
            acc += in_q0 * w_q0;
            if (i + 1 < in_dim) {
                acc += in_q1 * w_q1;
            }
        }
        
        output[o] = static_cast<float>(acc) * weight.scale * in_scale;
        
        if (bias) {
            int b_index = o / 2;
            uint8_t b_packed = bias->data[b_index];
            int8_t b_q = (o % 2 == 0) ? (b_packed & 0x0F) : ((b_packed >> 4) & 0x0F);
            if (b_q & 0x08) b_q |= 0xF0;
            output[o] += static_cast<float>(b_q) * bias->scale;
        }
    }
    
    total_mac_ops_per_token += (in_dim * out_dim);
    size_t current_act_bytes = output.size() * sizeof(float);
    if (current_act_bytes > max_activation_bytes) max_activation_bytes = current_act_bytes;
}

void layer_norm(std::vector<float>& x, const TensorQ4& gamma, const TensorQ4& beta) {
    float sum = 0.0f, sq_sum = 0.0f;
    for (float val : x) { sum += val; sq_sum += val * val; }
    float mean = sum / x.size();
    float var = (sq_sum / x.size()) - (mean * mean);
    float stddev = std::sqrt(var + 1e-5f);
    for (size_t i = 0; i < x.size(); ++i) {
        uint8_t g_packed = gamma.data[i / 2];
        int8_t g_q = (i % 2 == 0) ? (g_packed & 0x0F) : ((g_packed >> 4) & 0x0F);
        if (g_q & 0x08) g_q |= 0xF0;

        uint8_t b_packed = beta.data[i / 2];
        int8_t b_q = (i % 2 == 0) ? (b_packed & 0x0F) : ((b_packed >> 4) & 0x0F);
        if (b_q & 0x08) b_q |= 0xF0;

        float g = static_cast<float>(g_q) * gamma.scale;
        float b = static_cast<float>(b_q) * beta.scale;
        x[i] = ((x[i] - mean) / stddev) * g + b;
    }
}

void gelu(std::vector<float>& x) {
    for (float& val : x) {
        val = 0.5f * val * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (val + 0.044715f * val * val * val)));
    }
}

void softmax_attention(std::vector<float>& x) {
    if (x.empty()) return;
    float max_val = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;
    for (float& val : x) {
        val = std::exp(val - max_val);
        sum += val;
    }
    for (float& val : x) { val /= sum; }
}

// --- UTILIDADES ---

std::vector<std::string> load_vocabulary(const std::string& filename) {
    std::vector<std::string> vocab;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) { vocab.push_back(line); }
    return vocab;
}

std::string format_token(const std::string& t) {
    std::string out;
    for (size_t i = 0; i < t.length(); ) {
        if (i + 1 < t.length() && t[i] == '\xC4' && t[i+1] == '\xA0') {
            out += " "; i += 2;
        } else if (i + 1 < t.length() && t[i] == '\xC4' && t[i+1] == '\x8A') {
            out += "\n"; i += 2;
        } else {
            out += t[i]; i++;
        }
    }
    if (out == "<|endoftext|>") return "";
    return out;
}

TensorQ4 find_tensor(const std::vector<TensorQ4>& weights, const std::string& name) {
    for (const auto& w : weights) {
        if (w.name == name) return w;
    }
    std::cerr << "CRITICO: Falta el tensor " << name << std::endl;
    exit(1);
}

GPTNeoModel load_model(const std::string& filename, size_t& total_bytes) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) { std::cerr << "Error al abrir " << filename << std::endl; exit(1); }
    uint32_t magic; file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    
    std::vector<TensorQ4> all_weights;
    total_bytes = 0;
    
    while (file.peek() != EOF) {
        TensorQ4 t;
        uint32_t len;
        file.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (file.eof()) break;
        t.name.resize(len); file.read(&t.name[0], len);
        file.read(reinterpret_cast<char*>(&t.size), sizeof(t.size));
        file.read(reinterpret_cast<char*>(&t.scale), sizeof(t.scale));
        
        uint32_t packed_size = (t.size + 1) / 2;
        t.data.resize(packed_size);
        file.read(reinterpret_cast<char*>(t.data.data()), packed_size);
        
        total_bytes += packed_size; 
        all_weights.push_back(t);
    }

    GPTNeoModel model;
    model.wte = find_tensor(all_weights, "transformer.wte.weight");
    model.wpe = find_tensor(all_weights, "transformer.wpe.weight");
    
    for (int i = 0; i < NUM_LAYERS; ++i) {
        std::string prefix = "transformer.h." + std::to_string(i) + ".";
        Block b;
        b.ln_1_weight = find_tensor(all_weights, prefix + "ln_1.weight");
        b.ln_1_bias   = find_tensor(all_weights, prefix + "ln_1.bias");
        b.q_proj_w    = find_tensor(all_weights, prefix + "attn.attention.q_proj.weight");
        b.k_proj_w    = find_tensor(all_weights, prefix + "attn.attention.k_proj.weight");
        b.v_proj_w    = find_tensor(all_weights, prefix + "attn.attention.v_proj.weight");
        b.out_proj_w  = find_tensor(all_weights, prefix + "attn.attention.out_proj.weight");
        b.out_proj_b  = find_tensor(all_weights, prefix + "attn.attention.out_proj.bias");
        b.ln_2_weight = find_tensor(all_weights, prefix + "ln_2.weight");
        b.ln_2_bias   = find_tensor(all_weights, prefix + "ln_2.bias");
        b.mlp_fc_w    = find_tensor(all_weights, prefix + "mlp.c_fc.weight");
        b.mlp_fc_b    = find_tensor(all_weights, prefix + "mlp.c_fc.bias");
        b.mlp_proj_w  = find_tensor(all_weights, prefix + "mlp.c_proj.weight");
        b.mlp_proj_b  = find_tensor(all_weights, prefix + "mlp.c_proj.bias");
        model.blocks.push_back(b);
    }
    model.ln_f_weight = find_tensor(all_weights, "transformer.ln_f.weight");
    model.ln_f_bias = find_tensor(all_weights, "transformer.ln_f.bias");
    model.lm_head = find_tensor(all_weights, "lm_head.weight");
    
    return model;
}

// --- FORWARD PASS CON TOP-K Y ATENCION ---
int forward_token(GPTNeoModel& model, int token_id, int pos, KVCache& cache, const std::vector<int>& context_history) {
    total_mac_ops_per_token = 0; 
    std::vector<float> x(HIDDEN_SIZE, 0.0f);
    
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        uint8_t packed_wte = model.wte.data[(token_id * HIDDEN_SIZE + i) / 2];
        int8_t wte_q = ((token_id * HIDDEN_SIZE + i) % 2 == 0) ? (packed_wte & 0x0F) : (packed_wte >> 4);
        if (wte_q & 0x08) wte_q |= 0xF0;

        uint8_t packed_wpe = model.wpe.data[(pos * HIDDEN_SIZE + i) / 2];
        int8_t wpe_q = ((pos * HIDDEN_SIZE + i) % 2 == 0) ? (packed_wpe & 0x0F) : (packed_wpe >> 4);
        if (wpe_q & 0x08) wpe_q |= 0xF0;

        float t_emb = static_cast<float>(wte_q) * model.wte.scale;
        float p_emb = static_cast<float>(wpe_q) * model.wpe.scale;
        x[i] = t_emb + p_emb;
    }

    for (int l = 0; l < NUM_LAYERS; ++l) {
        const auto& block = model.blocks[l];
        auto& layer_cache = cache.layers[l];
        
        std::vector<float> residual = x;
        layer_norm(x, block.ln_1_weight, block.ln_1_bias);
        
        std::vector<float> q(HIDDEN_SIZE), k(HIDDEN_SIZE), v(HIDDEN_SIZE);
        linear_int4(x, block.q_proj_w, nullptr, q, HIDDEN_SIZE, HIDDEN_SIZE);
        linear_int4(x, block.k_proj_w, nullptr, k, HIDDEN_SIZE, HIDDEN_SIZE);
        linear_int4(x, block.v_proj_w, nullptr, v, HIDDEN_SIZE, HIDDEN_SIZE);
        
        layer_cache.k_past.push_back(k);
        layer_cache.v_past.push_back(v);
        
        int current_seq_len = layer_cache.k_past.size();
        std::vector<float> attn_out(HIDDEN_SIZE, 0.0f);
        
        for (int h = 0; h < NUM_HEADS; ++h) {
            int head_offset = h * HEAD_DIM;
            std::vector<float> scores(current_seq_len, 0.0f);
            
            for (int t_past = 0; t_past < current_seq_len; ++t_past) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += q[head_offset + d] * layer_cache.k_past[t_past][head_offset + d];
                }
                scores[t_past] = dot / std::sqrt(static_cast<float>(HEAD_DIM));
                total_mac_ops_per_token += HEAD_DIM; 
            }
            
            softmax_attention(scores);
            
            for (int t_past = 0; t_past < current_seq_len; ++t_past) {
                for (int d = 0; d < HEAD_DIM; ++d) {
                    attn_out[head_offset + d] += scores[t_past] * layer_cache.v_past[t_past][head_offset + d];
                    total_mac_ops_per_token += 1; 
                }
            }
        }
        
        std::vector<float> proj_out(HIDDEN_SIZE);
        linear_int4(attn_out, block.out_proj_w, &block.out_proj_b, proj_out, HIDDEN_SIZE, HIDDEN_SIZE);
        
        for (int i = 0; i < HIDDEN_SIZE; ++i) x[i] = residual[i] + proj_out[i];
        residual = x;
        
        layer_norm(x, block.ln_2_weight, block.ln_2_bias);
        std::vector<float> mlp_hidden(HIDDEN_SIZE * 4);
        linear_int4(x, block.mlp_fc_w, &block.mlp_fc_b, mlp_hidden, HIDDEN_SIZE, HIDDEN_SIZE * 4);
        gelu(mlp_hidden);
        
        std::vector<float> mlp_out(HIDDEN_SIZE);
        linear_int4(mlp_hidden, block.mlp_proj_w, &block.mlp_proj_b, mlp_out, HIDDEN_SIZE * 4, HIDDEN_SIZE);
        
        for (int i = 0; i < HIDDEN_SIZE; ++i) x[i] = residual[i] + mlp_out[i];
    }
    
    layer_norm(x, model.ln_f_weight, model.ln_f_bias);
    std::vector<float> logits(VOCAB_SIZE);
    linear_int4(x, model.lm_head, nullptr, logits, HIDDEN_SIZE, VOCAB_SIZE);
    
    float penalty = 1.2f; 
    for (int past_token : context_history) {
        if (logits[past_token] > 0) logits[past_token] /= penalty;
        else logits[past_token] *= penalty;
    }
    
    float temperature = 0.8f; 
    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(VOCAB_SIZE);
    float sum_probs = 0.0f;
    
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
        sum_probs += probs[i];
    }
    for (int i = 0; i < VOCAB_SIZE; ++i) probs[i] /= sum_probs;

    // --- FILTRO TOP-K ---
    int top_k = 200;
    std::vector<std::pair<float, int>> token_probs(VOCAB_SIZE);
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        token_probs[i] = {probs[i], i};
    }
    
    std::partial_sort(
        token_probs.begin(), 
        token_probs.begin() + top_k, 
        token_probs.end(), 
        [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first;
        }
    );
    
    std::vector<float> top_k_probs(top_k);
    float sum_top_k = 0.0f;
    for (int i = 0; i < top_k; ++i) {
        top_k_probs[i] = token_probs[i].first;
        sum_top_k += top_k_probs[i];
    }
    for (int i = 0; i < top_k; ++i) {
        top_k_probs[i] /= sum_top_k;
    }

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::discrete_distribution<> dist(top_k_probs.begin(), top_k_probs.end());
    
    int chosen_index = dist(gen);
    return token_probs[chosen_index].second;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  HARDWARE PROFILING INFO (TOP-K INT4)  " << std::endl;
    std::cout << "========================================" << std::endl;

    size_t total_weight_bytes = 0;
    auto t_start_load = std::chrono::high_resolution_clock::now();
    
    GPTNeoModel model = load_model("tinystories_1m_q4.bin", total_weight_bytes);
    std::vector<std::string> vocab = load_vocabulary("vocab.txt");
    
    auto t_end_load = std::chrono::high_resolution_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_end_load - t_start_load).count();

    std::cout << "[ROM/Flash] Memoria de Pesos INT4     : " << std::fixed << std::setprecision(2) << (total_weight_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "[INFO] Tiempo de carga en Host         : " << load_time << " ms\n" << std::endl;
    
    KVCache cache;
    std::vector<int> tokens = {7454, 2402, 257, 640}; 
    
    int next_token = 0;
    int max_new_tokens = 60; 
    
    std::cout << "--- FASE DE PREFILL (" << tokens.size() << " tokens) ---" << std::endl;
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::vector<int> current_context(tokens.begin(), tokens.begin() + i);
        next_token = forward_token(model, tokens[i], i, cache, current_context);
    }
    
    std::cout << "\n--- GENERACION AUTOREGRESIVA ---\n" << std::endl;
    
    for (int t : tokens) std::cout << format_token(vocab[t]);
    std::cout << std::flush;
    
    float total_gen_time = 0;
    int generated_count = 0;

    for (int step = 0; step < max_new_tokens; ++step) {
        if (next_token == END_OF_TEXT) break;
        
        std::cout << format_token(vocab[next_token]) << std::flush;
        tokens.push_back(next_token); 
        int current_pos = tokens.size() - 1;
        
        auto t_start_tok = std::chrono::high_resolution_clock::now();
        next_token = forward_token(model, next_token, current_pos, cache, tokens);
        auto t_end_tok = std::chrono::high_resolution_clock::now();
        
        total_gen_time += std::chrono::duration_cast<std::chrono::milliseconds>(t_end_tok - t_start_tok).count();
        generated_count++;
    }
    
    size_t kv_elements = tokens.size() * NUM_LAYERS * 2 * HIDDEN_SIZE;
    size_t kv_bytes_fp32 = kv_elements * 4;
    
    std::cout << "\n\n========================================" << std::endl;
    std::cout << "  REPORTE FINAL DE EJECUCION (TOP-K INT4)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "[BRAM] Peak Activations Buffer : " << max_activation_bytes << " Bytes" << std::endl;
    std::cout << "[BRAM] KV Cache Final Size     : " << kv_bytes_fp32 << " Bytes (en FP32)" << std::endl;
    std::cout << "[COMPUTE] MAC Ops por Token    : " << total_mac_ops_per_token << " operaciones" << std::endl;
    if (generated_count > 0) {
        std::cout << "[PERF] Velocidad Media Host    : " << (1000.0 / (total_gen_time / generated_count)) << " tok/s (" << (total_gen_time / generated_count) << " ms/tok)" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return 0;
}