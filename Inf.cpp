#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>

// 1. CAMBIO: Nueva estructura TensorQ4 usando uint8_t para bytes crudos empaquetados
struct TensorQ4 {
    std::string name;
    uint32_t size; // Cantidad REAL de parametros (ej. 50257 * 64)
    float scale;
    std::vector<uint8_t> data; // Arreglo empaquetado (pesara la mitad de 'size')
};

// Cargar el vocabulario desde vocab.txt
std::vector<std::string> load_vocabulary(const std::string& filename) {
    std::vector<std::string> vocab;
    std::ifstream file(filename);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Error al abrir " << filename << std::endl;
        return vocab;
    }
    
    while (std::getline(file, line)) {
        vocab.push_back(line);
    }
    return vocab;
}

int main() {
    // 1. Cargar Vocabulario
    std::vector<std::string> vocab = load_vocabulary("vocab.txt");
    std::cout << "Vocabulario cargado: " << vocab.size() << " tokens." << std::endl;

    // Asegurate de que el script de Python exportó con este nuevo nombre
    std::string bin_filename = "tinystories_1m_q4.bin"; 
    std::ifstream file(bin_filename, std::ios::binary);
    
    if (!file) {
        std::cerr << "Error: No se encontro " << bin_filename << std::endl;
        return 1;
    }
    
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x42494E31) {
        std::cerr << "Error: Magic number invalido." << std::endl;
        return 1;
    }
    
    std::vector<TensorQ4> model_weights;
    size_t total_memory_bytes = 0;
    
    std::cout << "\nCargando tensores desde disco..." << std::endl;
    
    while (file.peek() != EOF) {
        TensorQ4 tensor;
        uint32_t name_len;
        
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        if (file.eof()) break;
        
        tensor.name.resize(name_len);
        file.read(&tensor.name[0], name_len);
        file.read(reinterpret_cast<char*>(&tensor.size), sizeof(tensor.size));
        file.read(reinterpret_cast<char*>(&tensor.scale), sizeof(tensor.scale));
        
        // 2. CAMBIO: Calculamos los bytes reales a leer (dividimos entre 2 y redondeamos arriba)
        uint32_t packed_size = (tensor.size + 1) / 2;
        tensor.data.resize(packed_size);
        file.read(reinterpret_cast<char*>(tensor.data.data()), packed_size);
        
        total_memory_bytes += packed_size;
        model_weights.push_back(tensor);
    }
    
    std::cout << "Tensores cargados: " << model_weights.size() << std::endl;
    std::cout << "Memoria RAM consumida por pesos INT4: " << (total_memory_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    
    // 3. Prueba de De-cuantizacion (Golden Reference para Verilog)
    if (!model_weights.empty()) {
        TensorQ4& wte = model_weights[0];
        
        // 3. CAMBIO: Extraer y desempaquetar el primer byte
        uint8_t packed_byte = wte.data[0];
        
        // Tomamos los 4 bits menos significativos (el primer parametro)
        int8_t w_q0 = (packed_byte & 0x0F); 
        
        // Extensión de signo: si el bit 3 (valor 8) es 1, es un numero negativo en complemento a 2
        // Rellenamos el resto del byte con 1s para que C++ lo entienda como negativo
        if (w_q0 & 0x08) {
            w_q0 |= 0xF0; 
        }
        
        float real_value = static_cast<float>(w_q0) * wte.scale;
        
        std::cout << "\n--- TEST DE VERIFICACION MATEMATICA INT4 ---" << std::endl;
        std::cout << "Capa: " << wte.name << std::endl;
        std::cout << "Escala: " << wte.scale << std::endl;
        std::cout << "Byte empaquetado (HEX): 0x" << std::hex << static_cast<int>(packed_byte) << std::dec << std::endl;
        std::cout << "Valor INT4 extraido (indice 0): " << static_cast<int>(w_q0) << std::endl;
        std::cout << "Valor Float reconstruido: " << real_value << std::endl;
    }

    return 0;
}