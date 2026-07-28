"""
Re-cuantiza TinyStories-1M (GPT-Neo) al nuevo formato binario esperado
por el main.cpp corregido:

  - Matrices de pesos que participan en matmuls (embeddings, q/k/v/out_proj,
    mlp_fc, mlp_proj, lm_head): INT4 empaquetado, CON UN SCALE POR FILA
    (out_dim scales en vez de 1 global).
  - Bias y parametros de LayerNorm (ln_1, ln_2, ln_f, out_proj.bias,
    mlp.c_fc.bias, mlp.c_proj.bias): FP32 plano, sin cuantizar.

Requiere: pip install torch transformers

Formato de salida (tinystories_1m_q4.bin):
  uint32 magic
  por cada tensor:
    uint32 name_len; char name[name_len]
    uint8  type            (0 = Q4 por-fila, 1 = F32 plano)
    uint32 total_size       (cantidad de elementos)
    -- si type == 0:
         uint32 rows              (out_dim, cantidad de scales)
         float  scales[rows]
         uint8  data[ceil(total_size/2)]   (2 valores por byte, nibble bajo = indice par)
    -- si type == 1:
         float  data[total_size]
"""

import struct
import torch
from transformers import AutoModelForCausalLM

MAGIC = 0x54494E31  # "TIN1"

# Tensores que son matrices de pesos usadas en matmul -> cuantizar Q4 por fila.
# Todo lo demas (bias, layernorm) se guarda en F32 plano.
Q4_NAME_SUFFIXES = (
    "wte.weight",
    "wpe.weight",
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "out_proj.weight",
    "c_fc.weight",
    "c_proj.weight",
    "lm_head.weight",
)


def is_q4(name: str) -> bool:
    return any(name.endswith(suf) for suf in Q4_NAME_SUFFIXES)


def quantize_row_int4(row: torch.Tensor):
    """Cuantiza un vector 1D a INT4 con signo [-8, 7], devuelve (scale, valores_int8)."""
    max_abs = row.abs().max().item()
    scale = (max_abs / 7.0) if max_abs > 0 else 1.0
    q = torch.round(row / scale).clamp(-8, 7).to(torch.int8)
    return scale, q


def pack_int4(rows_q: torch.Tensor):
    """rows_q: tensor 2D [out_dim, in_dim] de valores int8 en [-8,7].
    Empaqueta 2 valores por byte, low-nibble = indice par, high-nibble = indice impar,
    recorriendo fila por fila (mismo layout que espera linear_int4 en el C++)."""
    flat = rows_q.flatten().tolist()
    packed = bytearray((len(flat) + 1) // 2)
    for idx, val in enumerate(flat):
        byte_idx = idx // 2
        nibble = val & 0x0F  # complemento a 2 de 4 bits
        if idx % 2 == 0:
            packed[byte_idx] = (packed[byte_idx] & 0xF0) | nibble
        else:
            packed[byte_idx] = (packed[byte_idx] & 0x0F) | (nibble << 4)
    return bytes(packed)


def write_q4_tensor(f, name: str, weight_2d: torch.Tensor):
    """weight_2d: [out_dim, in_dim] en FP32. Cuantiza CADA FILA con su propio scale."""
    out_dim, in_dim = weight_2d.shape
    scales = []
    quantized_rows = []
    for o in range(out_dim):
        scale, q_row = quantize_row_int4(weight_2d[o])
        scales.append(scale)
        quantized_rows.append(q_row)
    q_all = torch.stack(quantized_rows, dim=0)  # [out_dim, in_dim]
    packed = pack_int4(q_all)

    name_bytes = name.encode("utf-8")
    f.write(struct.pack("<I", len(name_bytes)))
    f.write(name_bytes)
    f.write(struct.pack("<B", 0))  # type = 0 (Q4)
    total_size = out_dim * in_dim
    f.write(struct.pack("<I", total_size))
    f.write(struct.pack("<I", out_dim))  # rows = cantidad de scales
    f.write(struct.pack(f"<{out_dim}f", *scales))
    f.write(packed)


def write_f32_tensor(f, name: str, tensor_1d: torch.Tensor):
    flat = tensor_1d.flatten().tolist()
    name_bytes = name.encode("utf-8")
    f.write(struct.pack("<I", len(name_bytes)))
    f.write(name_bytes)
    f.write(struct.pack("<B", 1))  # type = 1 (F32)
    f.write(struct.pack("<I", len(flat)))
    f.write(struct.pack(f"<{len(flat)}f", *flat))


def main():
    print("Cargando roneneldan/TinyStories-1M...")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
    state_dict = model.state_dict()

    out_path = "tinystories_1m_q4.bin"
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", MAGIC))

        for name, tensor in state_dict.items():
            tensor = tensor.detach().to(torch.float32)

            if is_q4(name):
                if tensor.dim() != 2:
                    raise ValueError(f"Se esperaba tensor 2D para {name}, tiene {tensor.dim()}D")
                write_q4_tensor(f, name, tensor)
                print(f"[Q4 ]  {name:50s} shape={tuple(tensor.shape)}")
            else:
                write_f32_tensor(f, name, tensor)
                print(f"[F32]  {name:50s} shape={tuple(tensor.shape)}")

    print(f"\nListo -> {out_path}")


if __name__ == "__main__":
    main()