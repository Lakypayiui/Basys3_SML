# SML-on-Basys3: A Small Language Model Running on an FPGA

## Project Goal

This project explores whether a **Small Language Model (SML)** — specifically a
GPT-Neo–architecture model in the TinyStories-1M family — can run **entirely
on-chip on a Digilent Basys 3 FPGA board**, generating text token-by-token
without any external host, GPU, or network connection.

The core constraint driving every design decision is **memory**. The Basys 3
was never built for machine learning workloads: its on-chip RAM is measured
in kilobytes, not megabytes, and its total non-volatile storage is a few
megabytes shared with the FPGA's own configuration bitstream. Making a
50M-class parameter language model fit and run at a usable speed inside that
budget is the central engineering problem of this project.

---

## Target Hardware: Digilent Basys 3

The Basys 3 is an entry-level FPGA trainer board built around a Xilinx
(AMD) Artix-7 FPGA. It was chosen for this project because it's an
accessible, low-cost platform for exploring embedded/edge inference — not
because it's well-suited for it out of the box.

| Component | Specification |
|---|---|
| FPGA | Xilinx/AMD Artix-7, part number `XC7A35T-1CPG236C` |
| Logic cells | 33,280 logic cells in 5,200 slices (4× 6-input LUTs + 8 flip-flops per slice) |
| Block RAM (BRAM) | 1,800 Kbits (~225 KB) of fast on-chip block RAM |
| DSP slices | 90 DSP slices (for multiply-accumulate hardware) |
| Clock management | 5 clock management tiles, each with a PLL |
| Internal clock speed | Exceeds 450 MHz (design-dependent) |
| Non-volatile storage | 32 Mbit (4 MB) Quad-SPI NOR Flash (Spansion S25FL032 or Macronix MX25L3233F, depending on unit) |
| Onboard I/O | 16 switches, 16 LEDs, 5 pushbuttons, 4-digit 7-segment display, 12-bit VGA, USB-UART bridge, USB HID host, 4 Pmod ports (3 standard 12-pin + 1 XADC/Pmod combo) |
| Toolchain | Xilinx/AMD Vivado Design Suite (WebPACK/Standard edition, free) |

### The real memory budget

The 4 MB Quad-SPI Flash is shared with the FPGA's own configuration
bitstream. An Artix-7 35T bitstream takes just over 2 MB, which leaves
**roughly 48% of the flash — under ~2 MB — actually free for user data**
such as model weights, once the design is deployed standalone (booting from
Flash rather than staying loaded via JTAG). This is a hard constraint the
project has to design around, on top of the more commonly cited "4 MB"
figure. During the C++ proof-of-concept phase weights are loaded from a
file on a host filesystem, so this limit isn't yet enforced — but it is the
real ceiling the Verilog implementation will have to hit.

The 1,800 Kbit (~225 KB) of BRAM is the other hard limit: it has to hold
activation buffers, the KV-cache, and any weights or lookup tables the
design chooses to keep resident on-chip during inference, all at once.

---

## Development Methodology: C++ First, Then Verilog

Given the tight iteration loop needed to get quantization, numerical
precision, and model behavior right, this project is being developed in
**two deliberate stages**:

### Stage 1 — C++ Proof of Concept (current stage)

A bit-accurate software model of the target hardware datapath is written in
C++ and run on a regular host machine (no FPGA involved yet). This lets the
architecture be validated and iterated on quickly:

- Model weights are quantized to **INT4**, packed 2 values per byte, mimicking
  the storage format the FPGA's Flash/BRAM will actually hold.
- All matrix multiplications are done with **integer accumulation** (INT4
  weights × dynamically-quantized INT8 activations), the same arithmetic a
  DSP-slice-based Verilog datapath would perform — not FP32 matmuls.
- Per-group weight scaling, LayerNorm/bias precision, KV-cache sizing, and
  attention/softmax behavior are all validated here first, where bugs are
  cheap to find and fix (recompile in seconds vs. a full Vivado
  synthesis/implementation run).
- The C++ code also acts as a **profiling harness**: it reports ROM/Flash
  footprint, peak activation (BRAM) usage, KV-cache size, MAC operations per
  token, and generation speed — all metrics that map directly onto Basys 3
  resource budgets (Flash capacity, BRAM capacity, DSP slice throughput).

The explicit intent is that **every numerical operation in the C++ code is
something that can be mapped 1:1 to a Verilog module** (a quantized linear
layer → a MAC/accumulator unit with a weight ROM; LayerNorm → a small
sequential arithmetic block; softmax/top-k → a comparator/sorting network),
so no rethinking of the algorithm is needed at the RTL stage — only
re-implementation of already-validated fixed-point arithmetic.

### Stage 2 — Verilog / RTL Port (planned)

Once the C++ model produces stable, coherent output within the target
memory budget, the datapath will be re-implemented in Verilog for synthesis
on the Basys 3:

- INT4 weights loaded from Quad-SPI Flash into on-chip BRAM (or streamed,
  depending on what fits).
- Quantized MAC operations mapped onto the Artix-7's 90 DSP slices.
- KV-cache and activation buffers implemented as BRAM-backed sequential
  logic.
- Control logic (layer sequencing, attention loop, top-k sampling) as a
  small FSM, replacing the C++ `for` loops.
- Token I/O via UART (leveraging the Basys 3's onboard USB-UART bridge) as
  the simplest way to get generated text off the board during bring-up,
  with the 7-segment display / LEDs as a fallback for basic status output.

The C++ version remains the reference model throughout Stage 2 — Verilog
simulation output is checked against it token-by-token to catch RTL bugs.

---

## Current Status (C++ Proof of Concept)

Latest run, after fixing a LayerNorm dequantization bug and moving to
per-group INT4 weight scaling (capped at 512 scale groups per tensor, to
keep the large embedding/output matrices affordable):

```
[ROM/Flash] Memoria de Pesos (Q4+F32) : 3.36 MB
[INFO] Tiempo de carga en Host         : 77 ms
--- FASE DE PREFILL (4 tokens) ---
--- GENERACION AUTOREGRESIVA ---
Once upon a time she in a big, very good friends: "Jenny decided to value. Then Jill asked eyes for better next old man for thank you is brave and happy voice. Suddenly Daddy going will his wish about it he special that looks fun."
That beautiful lber listened of the pictures could her family
========================================
  REPORTE FINAL DE EJECUCION (TOP-K INT4)
========================================
[BRAM] Peak Activations Buffer : 201028 Bytes
[BRAM] KV Cache Final Size     : 262144 Bytes (en FP32)
[COMPUTE] MAC Ops por Token    : 3675200 operaciones
[PERF] Velocidad Media Host    : 21.87 tok/s (45.72 ms/tok)
========================================
```

### Reading these numbers against the Basys 3 budget

| Metric | Value | Basys 3 constraint | Status |
|---|---|---|---|
| Weight storage (Q4 + F32) | 3.36 MB | ~2 MB free in Flash after bitstream | **Over budget** — needs further reduction |
| Peak activation buffer | 201,028 Bytes | ~225 KB total BRAM | Tight — leaves very little BRAM for KV-cache and control logic |
| KV-cache (FP32) | 262,144 Bytes | shares the same ~225 KB BRAM pool | **Over budget on its own** — will need INT8/INT4 KV-cache quantization |
| Generation speed (host) | ~21.9 tok/s | N/A (host, not FPGA) | Reference only — expect this to change significantly (likely slower per-token, but possibly deeply pipelined) once ported to RTL |

### Output quality

Text is now grammatically coherent in short spans — proper sentence
structure, punctuation, and dialogue formatting are present — but still
shows local corruption (`lber`, dropped/garbled words) consistent with
remaining INT4 quantization error, most likely concentrated in the large
`wte`/`lm_head` embedding matrices. This is being tracked as an open item
for the next quantization iteration, weighed against the Flash budget
above — every precision improvement here has a direct memory cost.

---

## Open Problems / Next Steps

1. **Flash budget**: 3.36 MB of weights vs. ~2 MB of realistically free
   Flash space is the single biggest blocker to a standalone (non-JTAG-tethered)
   deployment. Candidate mitigations: weight tying between `wte` and
   `lm_head` (if supported by this checkpoint, this alone would remove one
   of the two largest tensors), more aggressive group quantization, or
   accepting a smaller vocabulary.
2. **KV-cache size**: currently FP32; needs to move to INT8 or INT4 to fit
   inside the BRAM budget without starving activation buffers.
3. **Output quality**: continue isolating remaining quantization error,
   likely via targeted precision increases on the highest-impact tensors
   only (not a blanket precision increase, which the Flash budget can't
   absorb).
4. **RTL port**: begin translating the validated C++ datapath into Verilog
   once the above are resolved, starting with a single quantized linear
   layer as the first testable module.
