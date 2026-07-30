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

| Component | Specification |
|---|---|
| FPGA | Xilinx/AMD Artix-7, part number `XC7A35T-1CPG236C` |
| Logic cells | 33,280 logic cells in 5,200 slices (4× 6-input LUTs + 8 flip-flops per slice) |
| Block RAM (BRAM) | 1,800 Kbits (~225 KB) of fast on-chip block RAM |
| DSP slices | 90 DSP slices (for multiply-accumulate hardware) |
| Clock management | 5 clock management tiles, each with a PLL |
| Internal clock speed | Exceeds 450 MHz (design-dependent) |
| Non-volatile storage | 32 Mbit (4 MB) Quad-SPI NOR Flash (Spansion S25FL032 or Macronix MX25L3233F, depending on unit) |
| Onboard I/O | 16 switches, 16 LEDs, 5 pushbuttons, 4-digit 7-segment display, 12-bit VGA, USB-UART bridge, USB HID host, 4 Pmod ports |
| Toolchain | Xilinx/AMD Vivado Design Suite (WebPACK/Standard edition, free) |

### The real memory budget

The 4 MB Quad-SPI Flash is shared with the FPGA's own configuration
bitstream (~2 MB for an Artix-7 35T), leaving **roughly ~2 MB actually free
for user data** such as model weights in a standalone deployment (booting
from Flash rather than staying tethered to JTAG). This is the real ceiling
the Verilog implementation has to hit — not the commonly cited "4 MB"
figure.

The ~225 KB of BRAM is the other hard limit: it has to hold activation
buffers, the KV-cache, and any weights or lookup tables kept resident
on-chip during inference, all at once.

---

## Development Methodology: C++ First, Then Verilog

### Stage 1 — C++ Proof of Concept (complete for current scope)

A bit-accurate software model of the target hardware datapath, run on a
host machine, used to validate quantization and model behavior quickly
before committing to RTL. INT4 weights (packed 2-per-byte), integer
MAC accumulation, per-group weight scaling, and a full profiling harness
(ROM/Flash footprint, peak BRAM usage, KV-cache size, MAC ops/token,
tok/s) — all metrics that map directly onto Basys 3 resource budgets.

### Stage 2 — Verilog / RTL Port (in progress)

Translating the validated C++ datapath into Verilog module by module,
each validated in isolation before wiring into the full pipeline. This is
now the active stage of the project.

---

## Current Status

### C++ reference model — memory budget solved

Two optimizations closed the gap between the C++ model and the Basys 3's
real Flash/BRAM budgets, both **with zero quality cost** (unlike
quantization changes, which trade precision for size):

1. **Weight tying (`wte` ↔ `lm_head`)**: GPT-Neo ties the input embedding
   matrix and the output projection matrix by default — they are
   literally the same tensor. The original C++/`.bin` format stored both
   separately, duplicating ~1.5 MB of identical data. Storing it once:

   ```
   [ROM/Flash] Memoria de Pesos (Q4+F32) : 1.83 MB   (was 3.36 MB)
   ```

   This alone brought total weight storage from *over* the ~2 MB Flash
   budget to comfortably *under* it.

2. **KV-cache in FP16 instead of FP32**: halves KV-cache BRAM footprint
   with negligible precision loss for this use case:

   ```
   [BRAM] KV Cache Final Size : 131072 Bytes (en FP16)   (was 262144 in FP32)
   ```

Latest full run:

```
[ROM/Flash] Memoria de Pesos (Q4+F32) : 1.83 MB
[INFO] Tiempo de carga en Host         : 53 ms
--- FASE DE PREFILL (4 tokens) ---
--- GENERACION AUTOREGRESIVA ---
Once upon a time playing on an different singing and everyone could wiseer: maybe it it in value.
The little down night she was brave that happy that what new friend wantedily decided very good wish for him lucky friend never what bad happened to find honeyman right now he had anything fun important he found something wonderful
========================================
  REPORTE FINAL DE EJECUCION (TOP-K INT4)
========================================
[BRAM] Peak Activations Buffer : 201028 Bytes
[BRAM] KV Cache Final Size     : 131072 Bytes (en FP16)
[COMPUTE] MAC Ops por Token    : 3675200 operaciones
[PERF] Velocidad Media Host    : 21.17 tok/s (47.23 ms/tok)
========================================
```

#### Budget check against the Basys 3

| Metric | Value | Basys 3 constraint | Status |
|---|---|---|---|
| Weight storage (Q4 + F32, tied embeddings) | 1.83 MB | ~2 MB free in Flash |  **Under budget** |
| Peak activation buffer | 201,028 Bytes | ~225 KB total BRAM | Tight — still the largest single BRAM consumer (likely the 50257-entry logit vector; worth revisiting once the top-k module is built, e.g. an incremental top-k that never materializes the full vector) |
| KV-cache (FP16) | 131,072 Bytes | shares the same ~225 KB BRAM pool | Halved, but **201 KB + 131 KB together still exceed ~225 KB** — see RTL note below, this reappears in `attention.v`'s per-layer instance budget |
| Generation speed (host) | ~21.2 tok/s | N/A (host, not FPGA) | Reference only |

### Hardware bring-up — confirmed working on real Basys 3

The project has moved from "C++ only" to **real hardware verified on the
board**:

- **Menu / control FSM** (`top_fsm.v`): boot → connection banner → idle →
  prefill → generate → done → regenerate-or-exit, with debounced buttons
  (`btnC`/`btnU`/`btnD`) and a debounced physical reset (`btnR`).
- **UART link to the PC, confirmed working end-to-end**: the board sends
  `"Basys3 SML: connection OK"` over the USB-UART bridge (pin `A18`) at
  startup, visible in a serial terminal (PuTTY/screen, 115200 8N1) on the
  PC. Hit and fixed one real hardware bug along the way: an undebounced
  physical reset button was causing mid-transmission resets that
  corrupted the first UART bytes — same debounce pattern as the menu
  buttons fixed it.
- **`pc_listener.py`**: PC-side companion script. Works today in text
  passthrough mode (shows the connection banner and any status text).
  Also implements a token-ID mode (2-byte big-endian, vocab.txt lookup,
  same `format_token()` logic as the C++) as the **contract the future
  token-generation hardware needs to follow** — detokenization
  deliberately lives on the PC, not in scarce Flash/BRAM.

### RTL compute pipeline — modules built, arithmetic still pending

The compute datapath is being built module-by-module, each mapped
directly to a piece of the validated C++ reference:

| Module | Purpose | Status |
|---|---|---|
| `top_fsm.v` | Menu / control FSM |  Working on hardware |
| `uart_tx.v` | UART transmitter + fixed-string sender |  Working on hardware |
| `sml_top.v` | Top-level integration (menu + UART) |  Working on hardware |
| `flash_reader.v` | SPI Flash reader, **burst mode** (one header, many bytes) | Control logic complete; needs validation against Xilinx XAPP586 / a Digilent QSPI example before trusting the `STARTUPE2` usage on real hardware |
| `quant_linear.v` | Quantized linear layer (`linear_int4()` equivalent) | Control flow + Flash addressing complete (burst reads for weights/scale/bias, group-scale caching); **float arithmetic still `TODO`**, pending Vivado's Floating-Point Operator IP |
| `layer_norm.v` | LayerNorm, gamma/beta fetched from Flash in one burst each | Control flow complete; **float arithmetic + sqrt still `TODO`** |
| `attention.v` | Multi-head attention + FP16 KV-cache | Control flow + KV-cache addressing complete; **FP16↔FP32 conversion is fully implemented** (pure bit manipulation, no float IP needed); **score/softmax/weighted-sum arithmetic still `TODO`** |
| `topk_sampler.v` | Top-30 sampling over 50257 logits | Not yet built |
| `token_forward.v` | Orchestrates one full token through all of the above | Not yet built |

**Why the float arithmetic is still `TODO` everywhere**: comparing
magnitudes, multiplying by scales, and summing biases all need Vivado's
Floating-Point Operator IP (or an equivalent fixed-point redesign), and
that's a single, shared piece of work that touches `quant_linear.v`,
`layer_norm.v`, and `attention.v` simultaneously — deliberately deferred
until the control-flow skeletons of all three were in place, to connect
it once instead of three times.

### A known BRAM conflict, surfaced by `attention.v`'s design

`attention.v` needs its own KV-cache instance per transformer layer (all 8
layers' histories must coexist across the whole generation, not just the
current one). At `MAX_SEQ_LEN=128` that's 32 KB/layer × 8 layers = 256 KB
— already over the ~225 KB total BRAM budget **before** counting the
201 KB peak activation buffer. `MAX_SEQ_LEN=64` (the real max: 4 prefill +
60 generated tokens) roughly halves that to 128 KB, which is the direct
fix — flagged but not applied yet, pending a full BRAM budget pass once
`token_forward.v` makes the real per-module allocation visible.

---

## Open Problems / Next Steps

1. **Connect the Floating-Point Operator IP** across `quant_linear.v`,
   `layer_norm.v`, and `attention.v` — the single biggest remaining
   correctness gap between "control flow looks right" and "produces real
   numbers."
2. **`flash_reader.v` hardware validation**: confirm the `STARTUPE2`-based
   CCLK access against Xilinx's XAPP586 or a Digilent reference design
   before relying on it for real reads.
3. **BRAM budget pass**: resolve the `attention.v` KV-cache sizing
   (`MAX_SEQ_LEN` 128→64) and revisit the 201 KB peak activation buffer
   (likely the full logit vector — an incremental top-k that never
   materializes all 50257 at once would help here too).
4. **`topk_sampler.v`**: sequential top-30 scan + LFSR-based sampling over
   the (tied) `wte`/`lm_head` projection.
5. **`token_forward.v`**: the orchestrator tying every module above into
   one full token forward pass, replacing the placeholder counters
   currently driving `prefill_done`/`gen_done` in `sml_top.v`.
6. **Flash throughput**: even in burst mode, `flash_reader.v` is single-line
   SPI; Quad-SPI (4 data lines, ~4× throughput) is the next speed
   optimization once correctness is confirmed, and matters most for the
   tied `wte`/`lm_head` projection (50257 rows).
