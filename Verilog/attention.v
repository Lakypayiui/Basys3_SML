// =============================================================================
// attention.v
// -----------------------------------------------------------------------------
// Equivalente en RTL del bloque de atencion de forward_token() en el C++ de
// referencia. NO calcula q/k/v ni el out_proj -- esos son capas lineales
// normales, se resuelven reusando quant_linear.v (misma instancia, llamada
// 4 veces en secuencia por el futuro token_forward.v: q_proj, k_proj,
// v_proj, y despues out_proj sobre el resultado de este modulo). Este
// modulo SOLO hace: cachear k/v de este token, calcular los scores contra
// el historial, softmax, y la suma ponderada con v.
//
// El KV-cache se guarda en FP16 (uint16_t), igual que decidimos para el
// C++ -- por eso las funciones f32_to_f16 / f16_to_f32 de mas abajo son
// el port directo de las mismas funciones de main.cpp. A diferencia de
// TODA la aritmetica de punto flotante del resto del pipeline (que sigue
// marcada como TODO, pendiente del Floating-Point Operator IP de Vivado),
// estas dos conversiones SI estan completas: son manipulacion de bits pura,
// no necesitan ningun IP.
//
// Simplificacion intencional respecto al C++: f16_to_f32 no reconstruye
// denormales (los "flushea" a cero), igual que ya se discutio para el
// C++ -- irrelevante para las magnitudes tipicas de activaciones post-LN.
//
// KV-cache dimensionado para MAX_SEQ_LEN=128 posiciones (4 de prefill +
// hasta 60 generados, con margen). Memoria por instancia: 128*64*2(k,v)*2
// bytes = 32KB. Como hay NUM_LAYERS=8 capas y CADA UNA necesita su propio
// historial coexistiendo (no se puede reusar entre capas), token_forward.v
// va a instanciar 8 copias de este modulo -- 8*32KB=256KB, que es MAS que
// los ~225KB totales de BRAM del Basys3. Esto ya lo habiamos visto en el
// profiler del C++ (131072 Bytes con 64 tokens): con MAX_SEQ_LEN=128 este
// modulo reserva para el doble de contexto del que realmente se usa hoy.
// Bajar MAX_SEQ_LEN a 64 (el maximo real: 4 prefill + 60 generados) es la
// forma directa de que las 8 instancias entren en presupuesto -- lo dejo
// como parametro para decidir con el resto del sistema ya armado, no lo
// bajo yo unilateralmente aca.
// =============================================================================

module attention #(
    parameter HIDDEN_SIZE = 64,
    parameter NUM_HEADS   = 4,
    parameter HEAD_DIM    = HIDDEN_SIZE / NUM_HEADS,
    parameter MAX_SEQ_LEN = 128   // ver nota de presupuesto de BRAM arriba
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,

    // Cuantos tokens hay en el historial DESPUES de agregar este (incluye
    // el actual) -- equivalente a current_seq_len en el C++.
    input  wire [$clog2(MAX_SEQ_LEN)-1:0] seq_len,

    // q/k/v de ESTE token, ya calculados por quant_linear (BRAM, 1 ciclo)
    output reg  [$clog2(HIDDEN_SIZE)-1:0] q_addr,
    input  wire [31:0]                   q_data,
    output reg  [$clog2(HIDDEN_SIZE)-1:0] knew_addr,
    input  wire [31:0]                   knew_data,
    output reg  [$clog2(HIDDEN_SIZE)-1:0] vnew_addr,
    input  wire [31:0]                   vnew_data,

    // Salida: attn_out (HIDDEN_SIZE floats, BRAM), antes de pasar por out_proj
    output reg  [$clog2(HIDDEN_SIZE)-1:0] out_addr,
    output reg  [31:0]                    out_data,
    output reg                            out_we
);

    // -------------------------------------------------------------------
    // Conversion FP32 <-> FP16 (IEEE 754 binary16), port directo de
    // float_to_half()/half_to_float() en main.cpp. Combinacional, 0 ciclos
    // extra -- no necesitan el Floating-Point Operator IP.
    // -------------------------------------------------------------------
    function automatic [15:0] f32_to_f16(input [31:0] x);
        reg        sign_bit;
        reg [7:0]  raw_exp;
        reg [22:0] mant;
        reg signed [9:0] exp;
        begin
            sign_bit = x[31];
            raw_exp  = x[30:23];
            mant     = x[22:0];
            exp      = $signed({2'b00, raw_exp}) - 10'sd127 + 10'sd15;

            if (raw_exp == 8'hFF) begin
                f32_to_f16 = {sign_bit, 5'b11111, (mant != 0) ? 10'h200 : 10'h000};
            end else if (exp <= 0) begin
                f32_to_f16 = {sign_bit, 15'b0}; // flush a cero
            end else if (exp >= 31) begin
                f32_to_f16 = {sign_bit, 5'b11111, 10'b0}; // overflow -> inf
            end else begin
                f32_to_f16 = {sign_bit, exp[4:0], mant[22:13]};
            end
        end
    endfunction

    function automatic [31:0] f16_to_f32(input [15:0] h);
        reg        sign_bit;
        reg [4:0]  exp;
        reg [9:0]  mant;
        reg [7:0]  real_exp;
        begin
            sign_bit = h[15];
            exp      = h[14:10];
            mant     = h[9:0];
            real_exp = {3'b0, exp} + 8'd112; // -15 + 127

            if (exp == 5'd0) begin
                f16_to_f32 = {sign_bit, 31'b0}; // denormales -> 0 (ver nota arriba)
            end else if (exp == 5'd31) begin
                f16_to_f32 = {sign_bit, 8'hFF, mant, 13'b0};
            end else begin
                f16_to_f32 = {sign_bit, real_exp, mant, 13'b0};
            end
        end
    endfunction

    // -------------------------------------------------------------------
    // KV-cache (FP16). Direccion lineal: t_past * HIDDEN_SIZE + d
    // -------------------------------------------------------------------
    reg [15:0] k_cache_mem [0:MAX_SEQ_LEN*HIDDEN_SIZE-1];
    reg [15:0] v_cache_mem [0:MAX_SEQ_LEN*HIDDEN_SIZE-1];

    // Buffer de scores para la cabeza en curso (se reusa por cada head)
    reg [31:0] scores_mem [0:MAX_SEQ_LEN-1]; // float32 bits, aritmetica TODO

    // -------------------------------------------------------------------
    // Estados
    // -------------------------------------------------------------------
    localparam S_IDLE         = 4'd0;
    localparam S_CACHE_KV     = 4'd1;  // convertir y guardar k/v de este token
    localparam S_HEAD_INIT    = 4'd2;  // arrancar una cabeza nueva
    localparam S_SCORE_T_INIT = 4'd3;  // arrancar el dot product contra t_past
    localparam S_SCORE_D_LOOP = 4'd4;  // acumular sobre HEAD_DIM
    localparam S_SCORE_SAVE   = 4'd5;  // guardar el score escalado, siguiente t_past
    localparam S_SOFTMAX_MAX  = 4'd6;  // pasada 1: encontrar max (TODO fp_cmp)
    localparam S_SOFTMAX_EXP  = 4'd7;  // pasada 2: exp(x-max) y acumular suma (TODO)
    localparam S_SOFTMAX_NORM = 4'd8;  // pasada 3: dividir por la suma (TODO fp_div)
    localparam S_WSUM_D_INIT  = 4'd9;  // arrancar suma ponderada para columna d
    localparam S_WSUM_T_LOOP  = 4'd10; // acumular sobre t_past con v_cache
    localparam S_WSUM_WRITE   = 4'd11; // escribir attn_out[head_offset+d]
    localparam S_DONE         = 4'd12;

    reg [3:0] state;
    reg [$clog2(HIDDEN_SIZE):0]   cache_idx;
    reg [$clog2(NUM_HEADS):0]     h;
    reg [$clog2(MAX_SEQ_LEN):0]   t;
    reg [$clog2(HEAD_DIM):0]      d;
    reg [$clog2(HIDDEN_SIZE):0]   head_offset;

    reg signed [31:0] dot_acc;     // TODO: acumulador float real (fp_add/fp_mul)
    reg [31:0]        max_val;     // TODO: fp_cmp real
    reg [31:0]        sum_exp;     // TODO: fp_add real
    reg [31:0]        wsum_acc;    // TODO: acumulador float real

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            done      <= 1'b0;
            out_we    <= 1'b0;
            cache_idx <= 0;
            h         <= 0;
        end else begin
            done   <= 1'b0;
            out_we <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        cache_idx <= 0;
                        knew_addr <= 0;
                        vnew_addr <= 0;
                        state     <= S_CACHE_KV;
                    end
                end

                // Convertir k/v de este token (FP32 -> FP16) y guardarlos
                // en la posicion (seq_len-1) del cache -- el token actual
                // ya se cuenta dentro de seq_len, igual que en el C++
                // (push_back antes de leer current_seq_len).
                S_CACHE_KV: begin
                    k_cache_mem[(seq_len - 1) * HIDDEN_SIZE + cache_idx] <= f32_to_f16(knew_data);
                    v_cache_mem[(seq_len - 1) * HIDDEN_SIZE + cache_idx] <= f32_to_f16(vnew_data);

                    if (cache_idx == HIDDEN_SIZE - 1) begin
                        h     <= 0;
                        state <= S_HEAD_INIT;
                    end else begin
                        cache_idx <= cache_idx + 1'b1;
                        knew_addr <= knew_addr + 1'b1;
                        vnew_addr <= vnew_addr + 1'b1;
                    end
                end

                S_HEAD_INIT: begin
                    head_offset <= h * HEAD_DIM;
                    t           <= 0;
                    state       <= S_SCORE_T_INIT;
                end

                // Dot product q . k_cache[t] para esta cabeza
                S_SCORE_T_INIT: begin
                    dot_acc <= 32'sd0;
                    d       <= 0;
                    q_addr  <= head_offset;
                    state   <= S_SCORE_D_LOOP;
                end

                // TODO: dot_acc <= fp_add(dot_acc, fp_mul(q_data,
                //         f16_to_f32(k_cache_mem[t*HIDDEN_SIZE+head_offset+d])))
                S_SCORE_D_LOOP: begin
                    if (d == HEAD_DIM - 1) begin
                        state <= S_SCORE_SAVE;
                    end else begin
                        d      <= d + 1'b1;
                        q_addr <= q_addr + 1'b1;
                    end
                end

                // TODO: scores_mem[t] <= fp_div(dot_acc, sqrt(HEAD_DIM)) --
                // sqrt(HEAD_DIM) es una CONSTANTE de compilacion (HEAD_DIM=16
                // en este modelo -> sqrt(16)=4.0 exacto), no hace falta un
                // sqrt en tiempo real aca, solo una multiplicacion por la
                // constante 0.25 precalculada.
                S_SCORE_SAVE: begin
                    if (t == seq_len - 1) begin
                        state <= S_SOFTMAX_MAX;
                    end else begin
                        t     <= t + 1'b1;
                        state <= S_SCORE_T_INIT;
                    end
                end

                // TODO: recorrer scores_mem[0..seq_len-1], quedarse con el
                // maximo real (fp_cmp), igual que softmax_attention() en el C++.
                S_SOFTMAX_MAX: begin
                    t <= 0;
                    state <= S_SOFTMAX_EXP;
                end

                // TODO: scores_mem[t] <= exp(scores_mem[t] - max_val)
                // (LUT + interpolacion, o el IP de Vivado en modo exp),
                // sum_exp <= fp_add(sum_exp, scores_mem[t])
                S_SOFTMAX_EXP: begin
                    if (t == seq_len - 1) begin
                        t     <= 0;
                        state <= S_SOFTMAX_NORM;
                    end else begin
                        t <= t + 1'b1;
                    end
                end

                // TODO: scores_mem[t] <= fp_div(scores_mem[t], sum_exp)
                S_SOFTMAX_NORM: begin
                    if (t == seq_len - 1) begin
                        d     <= 0;
                        state <= S_WSUM_D_INIT;
                    end else begin
                        t <= t + 1'b1;
                    end
                end

                S_WSUM_D_INIT: begin
                    wsum_acc  <= 32'd0;
                    t         <= 0;
                    vnew_addr <= head_offset; // reusado como puntero de lectura, no de cache
                    state     <= S_WSUM_T_LOOP;
                end

                // TODO: wsum_acc <= fp_add(wsum_acc, fp_mul(scores_mem[t],
                //         f16_to_f32(v_cache_mem[t*HIDDEN_SIZE+head_offset+d])))
                S_WSUM_T_LOOP: begin
                    if (t == seq_len - 1) begin
                        state <= S_WSUM_WRITE;
                    end else begin
                        t <= t + 1'b1;
                    end
                end

                S_WSUM_WRITE: begin
                    out_addr <= head_offset + d;
                    out_data <= wsum_acc; // TODO: valor real via fp_add/fp_mul arriba
                    out_we   <= 1'b1;

                    if (d == HEAD_DIM - 1) begin
                        if (h == NUM_HEADS - 1) begin
                            state <= S_DONE;
                        end else begin
                            h     <= h + 1'b1;
                            state <= S_HEAD_INIT;
                        end
                    end else begin
                        d     <= d + 1'b1;
                        state <= S_WSUM_D_INIT;
                    end
                end

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule