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
#include <cstring>
#include <cstdint>

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

// --- FORMATO DE TENSORES ---
// TensorQ4: usado para matrices de pesos que participan en matmuls
// (embeddings, q/k/v/out_proj, mlp_fc, mlp_proj, lm_head).
// Tiene UN SCALE POR FILA (out_dim scales), no un scale global por tensor.
// Esto es la mejora de "cuantizacion por canal": cada fila de la matriz
// usa su propio rango dinamico en vez de compartir un solo scale que
// termina siendo arrastrado por los outliers de toda la matriz.
struct TensorQ4 {
    std::string name;
    uint32_t size;              // cantidad total de parametros (out_dim * in_dim)
    uint32_t out_dim;            // cantidad de filas de la matriz
    uint32_t group_size;         // cuantas filas consecutivas comparten un scale
    std::vector<float> scales;   // tamaño == ceil(out_dim / group_size)
    std::vector<uint8_t> data;   // 2 pesos por byte, empaquetados fila por fila

    // Para matrices chicas (q/k/v/out_proj, mlp: 64-256 filas) group_size=1,
    // o sea scale por fila real, igual que antes.
    // Para matrices enormes (wte/lm_head: 50257 filas) group_size>1, para no
    // pagar 50257 floats de overhead por un beneficio marginal fila-a-fila.
    inline float scale_for_row(uint32_t o) const { return scales[o / group_size]; }
};

// TensorF32: usado para bias y parametros de LayerNorm (gamma/beta).
// Son tensores chicos (decenas/cientos de valores) donde el ahorro de ROM
// al cuantizar es insignificante pero el error que meten en cada capa es
// grande porque afectan directamente la escala/offset de las activaciones.
// Se guardan en FP32 sin cuantizar.
struct TensorF32 {
    std::string name;
    std::vector<float> data;
};

struct Block {
    TensorF32 ln_1_weight, ln_1_bias;
    TensorQ4 q_proj_w, k_proj_w, v_proj_w;
    TensorQ4 out_proj_w;
    TensorF32 out_proj_b;
    TensorF32 ln_2_weight, ln_2_bias;
    TensorQ4 mlp_fc_w;
    TensorF32 mlp_fc_b;
    TensorQ4 mlp_proj_w;
    TensorF32 mlp_proj_b;
};

struct GPTNeoModel {
    TensorQ4 wte;
    TensorQ4 wpe;
    std::vector<Block> blocks;
    TensorF32 ln_f_weight, ln_f_bias;
    // No hay lm_head propio: esta tensor esta tied a wte (tie_word_embeddings=True
    // en GPT-Neo), asi que la proyeccion final reutiliza wte directamente. Ahorra
    // ~1.5MB de datos que serian identicos si se guardaran por separado.
};

struct LayerCache {
    std::vector<std::vector<uint16_t>> k_past; // FP16, no FP32
    std::vector<std::vector<uint16_t>> v_past; // FP16, no FP32
};

struct KVCache {
    std::vector<LayerCache> layers;
    KVCache() : layers(NUM_LAYERS) {}
};

// --- CONVERSION FP32 <-> FP16 (IEEE 754 binary16) ---
// Usado para el KV-cache: guardarlo en 16 bits en vez de 32 reduce a la mitad
// el BRAM que ocupa, y es ademas el formato natural para representar en
// Verilog (un registro de 16 bits) en la futura migracion a RTL.

uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;

    if (((x >> 23) & 0xFF) == 0xFF) {
        // Inf o NaN
        return static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x200 : 0));
    }
    if (exp <= 0) {
        // Demasiado chico para un half normalizado -> flush to zero
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        // Overflow -> infinito
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float half_to_float(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign; // cero
        } else {
            // denormal de half -> normalizar
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13); // inf/nan
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }

    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

// --- OPERACIONES MATEMATICAS FUNDAMENTALES (INT4 por fila) ---

void linear_int4(const std::vector<float>& input, const TensorQ4& weight, const TensorF32* bias, std::vector<float>& output, int in_dim, int out_dim) {
    float max_in = 0.0f;
    for (float val : input) {
        if (std::abs(val) > max_in) max_in = std::abs(val);
    }
    float in_scale = max_in > 0 ? max_in / 127.0f : 1.0f;

    for (int o = 0; o < out_dim; ++o) {
        int32_t acc = 0;
        float w_scale = weight.scale_for_row(o); // scale del grupo al que pertenece esta fila

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

        output[o] = static_cast<float>(acc) * w_scale * in_scale;

        if (bias) {
            output[o] += bias->data[o];
        }
    }

    total_mac_ops_per_token += (in_dim * out_dim);
    size_t current_act_bytes = output.size() * sizeof(float);
    if (current_act_bytes > max_activation_bytes) max_activation_bytes = current_act_bytes;
}

void layer_norm(std::vector<float>& x, const TensorF32& gamma, const TensorF32& beta) {
    float sum = 0.0f, sq_sum = 0.0f;
    for (float val : x) { sum += val; sq_sum += val * val; }
    float mean = sum / x.size();
    float var = (sq_sum / x.size()) - (mean * mean);
    float stddev = std::sqrt(var + 1e-5f);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = ((x[i] - mean) / stddev) * gamma.data[i] + beta.data[i];
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

TensorQ4 find_q4_tensor(const std::vector<TensorQ4>& weights, const std::string& name) {
    for (const auto& w : weights) {
        if (w.name == name) return w;
    }
    std::cerr << "CRITICO: Falta el tensor Q4 " << name << std::endl;
    exit(1);
}

TensorF32 find_f32_tensor(const std::vector<TensorF32>& weights, const std::string& name) {
    for (const auto& w : weights) {
        if (w.name == name) return w;
    }
    std::cerr << "CRITICO: Falta el tensor F32 " << name << std::endl;
    exit(1);
}

// --- FORMATO DEL .bin ---
// Por cada tensor:
//   uint32 name_len; char name[name_len]
//   uint8  type            (0 = Q4 por-fila, 1 = F32 plano)
//   uint32 total_size       (cantidad de elementos)
//   -- si type == 0 (Q4):
//        uint32 out_dim            (cantidad de filas de la matriz)
//        uint32 group_size         (filas consecutivas por scale)
//        float  scales[ceil(out_dim / group_size)]
//        uint8  data[ceil(total_size/2)]   (packed, 2 valores por byte)
//   -- si type == 1 (F32):
//        float  data[total_size]
GPTNeoModel load_model(const std::string& filename, size_t& total_bytes) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) { std::cerr << "Error al abrir " << filename << std::endl; exit(1); }
    uint32_t magic; file.read(reinterpret_cast<char*>(&magic), sizeof(magic));

    std::vector<TensorQ4> q4_weights;
    std::vector<TensorF32> f32_weights;
    total_bytes = 0;

    while (file.peek() != EOF) {
        uint32_t len;
        file.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (file.eof()) break;

        std::string name;
        name.resize(len);
        file.read(&name[0], len);

        uint8_t type;
        file.read(reinterpret_cast<char*>(&type), sizeof(type));

        uint32_t total_size;
        file.read(reinterpret_cast<char*>(&total_size), sizeof(total_size));

        if (type == 0) {
            TensorQ4 t;
            t.name = name;
            t.size = total_size;

            file.read(reinterpret_cast<char*>(&t.out_dim), sizeof(t.out_dim));
            file.read(reinterpret_cast<char*>(&t.group_size), sizeof(t.group_size));

            uint32_t num_groups = (t.out_dim + t.group_size - 1) / t.group_size;
            t.scales.resize(num_groups);
            file.read(reinterpret_cast<char*>(t.scales.data()), num_groups * sizeof(float));

            uint32_t packed_size = (t.size + 1) / 2;
            t.data.resize(packed_size);
            file.read(reinterpret_cast<char*>(t.data.data()), packed_size);

            total_bytes += packed_size + num_groups * sizeof(float);
            q4_weights.push_back(t);
        } else {
            TensorF32 t;
            t.name = name;
            t.data.resize(total_size);
            file.read(reinterpret_cast<char*>(t.data.data()), total_size * sizeof(float));

            total_bytes += total_size * sizeof(float);
            f32_weights.push_back(t);
        }
    }

    GPTNeoModel model;
    model.wte = find_q4_tensor(q4_weights, "transformer.wte.weight");
    model.wpe = find_q4_tensor(q4_weights, "transformer.wpe.weight");

    for (int i = 0; i < NUM_LAYERS; ++i) {
        std::string prefix = "transformer.h." + std::to_string(i) + ".";
        Block b;
        b.ln_1_weight = find_f32_tensor(f32_weights, prefix + "ln_1.weight");
        b.ln_1_bias   = find_f32_tensor(f32_weights, prefix + "ln_1.bias");
        b.q_proj_w    = find_q4_tensor(q4_weights, prefix + "attn.attention.q_proj.weight");
        b.k_proj_w    = find_q4_tensor(q4_weights, prefix + "attn.attention.k_proj.weight");
        b.v_proj_w    = find_q4_tensor(q4_weights, prefix + "attn.attention.v_proj.weight");
        b.out_proj_w  = find_q4_tensor(q4_weights, prefix + "attn.attention.out_proj.weight");
        b.out_proj_b  = find_f32_tensor(f32_weights, prefix + "attn.attention.out_proj.bias");
        b.ln_2_weight = find_f32_tensor(f32_weights, prefix + "ln_2.weight");
        b.ln_2_bias   = find_f32_tensor(f32_weights, prefix + "ln_2.bias");
        b.mlp_fc_w    = find_q4_tensor(q4_weights, prefix + "mlp.c_fc.weight");
        b.mlp_fc_b    = find_f32_tensor(f32_weights, prefix + "mlp.c_fc.bias");
        b.mlp_proj_w  = find_q4_tensor(q4_weights, prefix + "mlp.c_proj.weight");
        b.mlp_proj_b  = find_f32_tensor(f32_weights, prefix + "mlp.c_proj.bias");
        model.blocks.push_back(b);
    }
    model.ln_f_weight = find_f32_tensor(f32_weights, "transformer.ln_f.weight");
    model.ln_f_bias = find_f32_tensor(f32_weights, "transformer.ln_f.bias");
    // lm_head.weight no se carga: esta tied a wte, se reutiliza model.wte
    // directamente en el forward pass (ver forward_token).

    return model;
}

// --- FORWARD PASS CON TOP-K Y ATENCION ---
int forward_token(GPTNeoModel& model, int token_id, int pos, KVCache& cache, const std::vector<int>& context_history) {
    total_mac_ops_per_token = 0;
    std::vector<float> x(HIDDEN_SIZE, 0.0f);

    // wte y wpe usan scale por grupo (grupo = varios tokens/posiciones consecutivas
    // compartiendo un scale, para no pagar 50257 floats de overhead en wte)
    float wte_scale = model.wte.scale_for_row(token_id);
    float wpe_scale = model.wpe.scale_for_row(pos);

    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        uint8_t packed_wte = model.wte.data[(token_id * HIDDEN_SIZE + i) / 2];
        int8_t wte_q = ((token_id * HIDDEN_SIZE + i) % 2 == 0) ? (packed_wte & 0x0F) : (packed_wte >> 4);
        if (wte_q & 0x08) wte_q |= 0xF0;

        uint8_t packed_wpe = model.wpe.data[(pos * HIDDEN_SIZE + i) / 2];
        int8_t wpe_q = ((pos * HIDDEN_SIZE + i) % 2 == 0) ? (packed_wpe & 0x0F) : (packed_wpe >> 4);
        if (wpe_q & 0x08) wpe_q |= 0xF0;

        float t_emb = static_cast<float>(wte_q) * wte_scale;
        float p_emb = static_cast<float>(wpe_q) * wpe_scale;
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

        // El cache guarda K/V en FP16: convertimos antes de empujar.
        std::vector<uint16_t> k_half(HIDDEN_SIZE), v_half(HIDDEN_SIZE);
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            k_half[i] = float_to_half(k[i]);
            v_half[i] = float_to_half(v[i]);
        }
        layer_cache.k_past.push_back(k_half);
        layer_cache.v_past.push_back(v_half);

        int current_seq_len = layer_cache.k_past.size();
        std::vector<float> attn_out(HIDDEN_SIZE, 0.0f);

        for (int h = 0; h < NUM_HEADS; ++h) {
            int head_offset = h * HEAD_DIM;
            std::vector<float> scores(current_seq_len, 0.0f);

            for (int t_past = 0; t_past < current_seq_len; ++t_past) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += q[head_offset + d] * half_to_float(layer_cache.k_past[t_past][head_offset + d]);
                }
                scores[t_past] = dot / std::sqrt(static_cast<float>(HEAD_DIM));
                total_mac_ops_per_token += HEAD_DIM;
            }

            softmax_attention(scores);

            for (int t_past = 0; t_past < current_seq_len; ++t_past) {
                for (int d = 0; d < HEAD_DIM; ++d) {
                    attn_out[head_offset + d] += scores[t_past] * half_to_float(layer_cache.v_past[t_past][head_offset + d]);
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
    // lm_head esta tied a wte: la misma matriz [VOCAB_SIZE, HIDDEN_SIZE] que se usa
    // para el embedding de entrada se reutiliza aca como matriz de proyeccion de salida.
    linear_int4(x, model.wte, nullptr, logits, HIDDEN_SIZE, VOCAB_SIZE);

    float penalty = 1.2f;
    for (int past_token : context_history) {
        if (logits[past_token] > 0) logits[past_token] /= penalty;
        else logits[past_token] *= penalty;
    }

    float temperature = 0.7f;
    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(VOCAB_SIZE);
    float sum_probs = 0.0f;

    for (int i = 0; i < VOCAB_SIZE; ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
        sum_probs += probs[i];
    }
    for (int i = 0; i < VOCAB_SIZE; ++i) probs[i] /= sum_probs;

    // --- FILTRO TOP-K ---
    int top_k = 30;
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

    std::cout << "[ROM/Flash] Memoria de Pesos (Q4+F32) : " << std::fixed << std::setprecision(2) << (total_weight_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
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
    size_t kv_bytes_fp16 = kv_elements * sizeof(uint16_t);

    std::cout << "\n\n========================================" << std::endl;
    std::cout << "  REPORTE FINAL DE EJECUCION (TOP-K INT4)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "[BRAM] Peak Activations Buffer : " << max_activation_bytes << " Bytes" << std::endl;
    std::cout << "[BRAM] KV Cache Final Size     : " << kv_bytes_fp16 << " Bytes (en FP16)" << std::endl;
    std::cout << "[COMPUTE] MAC Ops por Token    : " << total_mac_ops_per_token << " operaciones" << std::endl;
    if (generated_count > 0) {
        std::cout << "[PERF] Velocidad Media Host    : " << (1000.0 / (total_gen_time / generated_count)) << " tok/s (" << (total_gen_time / generated_count) << " ms/tok)" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return 0;
}