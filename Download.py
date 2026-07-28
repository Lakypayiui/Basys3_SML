import torch
import struct
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def export_model_and_vocab():
    model_name = "roneneldan/TinyStories-1M"
    tokenizer_name = "EleutherAI/gpt-neo-125M" # Base usada para TinyStories
    
    print(f"1. Descargando el tokenizador desde Hugging Face...")
    # El tokenizador se descarga y cachea localmente
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    print("2. Exportando vocabulario a vocab.txt para C++...")
    vocab = tokenizer.get_vocab()
    # Ordenar por el ID del token para que el indice coincida con la linea del archivo
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    with open("vocab.txt", "w", encoding="utf-8") as f:
        for word, idx in sorted_vocab:
            # Limpiamos saltos de linea nativos para no romper el parser en C++
            clean_word = word.replace('\n', '\\n').replace('\r', '\\r')
            f.write(f"{clean_word}\n")
    print(f"   -> Vocabulario guardado ({len(sorted_vocab)} tokens).")

    print(f"\n3. Descargando los pesos de {model_name}...")
    # Descarga de la red neuronal a la memoria RAM
    model = AutoModelForCausalLM.from_pretrained(model_name)
    state_dict = model.state_dict()
    
    output_file = "tinystories_1m_q4.bin"
    print(f"\n4. Iniciando cuantizacion INT4 y empaquetado binario...")
    
    with open(output_file, 'wb') as f:
        # Magic number para que C++ sepa que esta leyendo el archivo correcto
        f.write(struct.pack('I', 0x42494E31)) 
        
        total_params = 0
        total_bytes = 0
        
        for name, tensor in state_dict.items():
            # Convertir a float32 y aplanar el tensor a 1D
            w = tensor.detach().cpu().float().numpy().flatten()
            total_params += w.size
            
            # 1. Calcular factor de escala para INT4 (rango -8 a 7)
            max_val = max(abs(w.max()), abs(w.min()))
            scale = max_val / 7.0 if max_val > 0 else 1.0
            
            # 2. Cuantizar a INT4 y castear a int8 temporalmente
            w_int4 = (w / scale).round().clip(-8, 7).astype(np.int8)
            
            # Si el tensor es impar, agregamos un 0 de relleno para poder empaquetar en pares
            if w_int4.size % 2 != 0:
                w_int4 = np.append(w_int4, [0]).astype(np.int8)
            
            # 3. Empaquetar 2 valores INT4 en 1 byte (uint8)
            # Aplicamos mascara 0x0F (00001111) para limpiar los bits de signo en negativos
            val0 = w_int4[0::2] & 0x0F
            val1 = w_int4[1::2] & 0x0F
            
            # Bitshift de 4 posiciones para el segundo valor y operacion OR
            w_packed = np.bitwise_or(val0, np.left_shift(val1, 4)).astype(np.uint8)
            
            # 4. Guardar en el binario:
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))    
            f.write(name_bytes)                           
            f.write(struct.pack('I', w.size)) # Mantenemos el tamano original de parametros
            f.write(struct.pack('f', scale))              
            f.write(w_packed.tobytes()) # Escribimos la mitad de los bytes
            
            total_bytes += (4 + len(name_bytes) + 4 + 4 + w_packed.nbytes)
            print(f"   -> Capa: {name: <40} | Escala: {scale:.6f} | Bytes: {w_packed.nbytes}")

    print("-" * 50)
    print(f"EXITO: Procesamiento completado.")
    print(f"Parametros empaquetados: {total_params:,}")
    print(f"Tamano final en disco:   {total_bytes / (1024*1024):.2f} MB")
    print(f"Archivos generados:      {output_file}, vocab.txt")

if __name__ == "__main__":
    export_model_and_vocab()