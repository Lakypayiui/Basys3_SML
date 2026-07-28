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

# Tope maximo de grupos de scale por tensor. Con esto una matriz de 50257 filas
# (wte, lm_head) no paga 50257 floats de scale -- paga a lo sumo MAX_GROUPS.
# Matrices chicas (64-256 filas, ya por debajo del tope) siguen teniendo scale
# por fila real, sin cambios respecto a la version anterior.
#
# Trade-off: bajar este numero ahorra mas ROM pero pierde granularidad de
# cuantizacion en wte/lm_head (vuelve a acercarse al bug original de scale
# global mientras mas chico sea). 512 es un punto de partida razonable;
# si necesitas exprimir mas MB, probar 128 o 256 antes de ir mas abajo.
MAX_GROUPS = 512

# Tensores que son matrices de pesos usadas en matmul -> cuantizar Q4 por grupo.
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


def choose_group_size(out_dim: int, max_groups: int = MAX_GROUPS) -> int:
    """Cuantas filas consecutivas comparten un scale, para no superar max_groups
    scales en total por tensor."""
    num_groups = min(out_dim, max_groups)
    return (out_dim + num_groups - 1) // num_groups


def quantize_group_int4(rows: torch.Tensor):
    """rows: [group_size, in_dim]. Cuantiza el grupo entero a INT4 con signo
    [-8, 7] usando un unico scale para todas esas filas juntas."""
    max_abs = rows.abs().max().item()
    scale = (max_abs / 7.0) if max_abs > 0 else 1.0
    q = torch.round(rows / scale).clamp(-8, 7).to(torch.int8)
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
    """weight_2d: [out_dim, in_dim] en FP32. Cuantiza en grupos de
    `group_size` filas consecutivas, cada grupo con su propio scale,
    respetando el tope MAX_GROUPS scales por tensor."""
    out_dim, in_dim = weight_2d.shape
    group_size = choose_group_size(out_dim)

    scales = []
    quantized_rows = [None] * out_dim
    for start in range(0, out_dim, group_size):
        end = min(start + group_size, out_dim)
        scale, q_group = quantize_group_int4(weight_2d[start:end])
        scales.append(scale)
        for offset, row_idx in enumerate(range(start, end)):
            quantized_rows[row_idx] = q_group[offset]

    q_all = torch.stack(quantized_rows, dim=0)  # [out_dim, in_dim], orden original
    packed = pack_int4(q_all)

    name_bytes = name.encode("utf-8")
    f.write(struct.pack("<I", len(name_bytes)))
    f.write(name_bytes)
    f.write(struct.pack("<B", 0))  # type = 0 (Q4)
    total_size = out_dim * in_dim
    f.write(struct.pack("<I", total_size))
    f.write(struct.pack("<I", out_dim))
    f.write(struct.pack("<I", group_size))
    f.write(struct.pack(f"<{len(scales)}f", *scales))
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
                out_dim = tensor.shape[0]
                gsize = choose_group_size(out_dim)
                num_groups = (out_dim + gsize - 1) // gsize
                write_q4_tensor(f, name, tensor)
                print(f"[Q4 ]  {name:50s} shape={tuple(tensor.shape)}  group_size={gsize} num_groups={num_groups}")
            else:
                write_f32_tensor(f, name, tensor)
                print(f"[F32]  {name:50s} shape={tuple(tensor.shape)}")

    print(f"\nListo -> {out_path}")


if __name__ == "__main__":
    main()