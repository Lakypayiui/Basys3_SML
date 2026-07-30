// =============================================================================
// layer_norm.v
// -----------------------------------------------------------------------------
// Equivalente en RTL de layer_norm() en el C++ de referencia. Recorre las
// DIM activaciones dos veces (igual que el C++): una para acumular sum y
// sum-of-squares, otra para aplicar la normalizacion con gamma/beta.
//
// gamma y beta viven en Flash como tensores F32 planos (no cuantizados,
// misma decision que ya tomamos para bias/LN en quantize.py), y se traen
// UNA vez al arrancar, en modo rafaga, a un array chico en fabric (DIM
// floats cada uno -- 64 en este modelo, perfectamente manejable en
// registros; si DIM creciera mucho convendria pasar esto a BRAM en vez de
// flip-flops, pero para HIDDEN_SIZE=64 no hace falta).
//
// La aritmetica de punto flotante (suma, resta, multiplicacion, division,
// RAIZ CUADRADA) sigue marcada como TODO, igual que en quant_linear.v --
// la raiz cuadrada en particular no tiene un IP tan directo como suma/mult;
// Vivado la resuelve con el Floating-Point Operator IP en modo sqrt
// (Newton-Raphson internamente), o hay que implementarla a mano con CORDIC
// si no se quiere pagar el costo de ese IP.
// =============================================================================

module layer_norm #(
    parameter DIM = 64
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,

    // Activaciones de entrada/salida (BRAM, 1 ciclo). Se lee DOS veces
    // (una para stats, otra para normalizar), y se escribe una vez.
    output reg  [$clog2(DIM)-1:0] in_addr,
    input  wire [31:0]            in_data,
    output reg  [$clog2(DIM)-1:0] out_addr,
    output reg  [31:0]            out_data,
    output reg                    out_we,

    // Direcciones base en Flash de gamma y beta para esta capa (ln_1, ln_2
    // o ln_f -- las tres usan esta misma instancia, solo cambian las bases)
    input  wire [23:0] gamma_base,
    input  wire [23:0] beta_base,

    // Bus compartido hacia flash_reader.v v2 (modo rafaga)
    output reg  [23:0] flash_addr,
    output reg  [15:0] flash_burst_len,
    output reg         flash_start,
    input  wire        flash_busy,
    input  wire [7:0]  flash_data,
    input  wire        flash_valid,
    input  wire        flash_done
);

    // Cache local de gamma/beta (DIM floats cada uno). Para DIM=64 son 64
    // registros de 32 bits por array -- razonable en fabric; si DIM creciera
    // mucho, pasar esto a un BRAM chico en vez de flip-flops.
    reg [31:0] gamma_arr [0:DIM-1];
    reg [31:0] beta_arr  [0:DIM-1];

    localparam S_IDLE        = 4'd0;
    localparam S_SCAN        = 4'd1;   // acumular sum y sum-of-squares (BRAM, rapido)
    localparam S_GAMMA_START = 4'd2;   // lanzar rafaga de gamma
    localparam S_GAMMA_WAIT  = 4'd3;   // recibir los DIM*4 bytes de gamma
    localparam S_BETA_START  = 4'd4;   // lanzar rafaga de beta
    localparam S_BETA_WAIT   = 4'd5;   // recibir los DIM*4 bytes de beta
    localparam S_STATS       = 4'd6;   // mean, var, stddev (TODO: fp_div/fp_sqrt)
    localparam S_NORM        = 4'd7;   // segunda pasada: aplicar gamma/beta
    localparam S_DONE        = 4'd8;

    reg [3:0] state;
    reg [$clog2(DIM):0] idx;       // indice generico (scan / norm)
    reg [$clog2(DIM):0] elem_idx;  // indice de elemento durante rafagas gamma/beta
    reg [1:0] byte_in_float;       // 0..3, byte actual dentro del float en curso
    reg [31:0] float_accum;        // ensamblado little-endian, igual que <f de Python

    reg [31:0] sum_reg;      // TODO: acumulador float real (fp_add)
    reg [31:0] sqsum_reg;    // TODO: acumulador float real (fp_add + fp_mul)
    reg [31:0] mean_reg;     // TODO: sum_reg / DIM (fp_div, o *reciproco precalculado)
    reg [31:0] var_reg;      // TODO: sqsum_reg/DIM - mean_reg*mean_reg
    reg [31:0] stddev_reg;   // TODO: sqrt(var_reg + eps) (fp_sqrt)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            done            <= 1'b0;
            out_we          <= 1'b0;
            idx             <= 0;
            elem_idx        <= 0;
            byte_in_float   <= 2'd0;
            sum_reg         <= 32'd0;
            sqsum_reg       <= 32'd0;
            flash_start     <= 1'b0;
        end else begin
            done        <= 1'b0;
            out_we      <= 1'b0;
            flash_start <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        idx       <= 0;
                        in_addr   <= 0;
                        sum_reg   <= 32'd0;
                        sqsum_reg <= 32'd0;
                        state     <= S_SCAN;
                    end
                end

                // Primera pasada: sum y sum-of-squares (BRAM, 1 ciclo por
                // elemento, no toca Flash).
                // TODO: sum_reg <= fp_add(sum_reg, in_data)
                //       sqsum_reg <= fp_add(sqsum_reg, fp_mul(in_data, in_data))
                S_SCAN: begin
                    if (idx == DIM - 1) begin
                        elem_idx        <= 0;
                        byte_in_float   <= 2'd0;
                        flash_addr      <= gamma_base;
                        flash_burst_len <= DIM * 4;
                        flash_start     <= 1'b1;
                        state           <= S_GAMMA_START;
                    end else begin
                        idx     <= idx + 1'b1;
                        in_addr <= in_addr + 1'b1;
                    end
                end

                S_GAMMA_START: begin
                    if (flash_busy) state <= S_GAMMA_WAIT; // la rafaga ya arranco
                end

                S_GAMMA_WAIT: begin
                    if (flash_valid) begin
                        float_accum <= {flash_data, float_accum[31:8]};
                        if (byte_in_float == 2'd3) begin
                            gamma_arr[elem_idx] <= {flash_data, float_accum[31:8]};
                            byte_in_float       <= 2'd0;
                            elem_idx            <= elem_idx + 1'b1;
                        end else begin
                            byte_in_float <= byte_in_float + 1'b1;
                        end
                    end
                    if (flash_done) begin
                        elem_idx        <= 0;
                        byte_in_float   <= 2'd0;
                        flash_addr      <= beta_base;
                        flash_burst_len <= DIM * 4;
                        flash_start     <= 1'b1;
                        state           <= S_BETA_START;
                    end
                end

                S_BETA_START: begin
                    if (flash_busy) state <= S_BETA_WAIT;
                end

                S_BETA_WAIT: begin
                    if (flash_valid) begin
                        float_accum <= {flash_data, float_accum[31:8]};
                        if (byte_in_float == 2'd3) begin
                            beta_arr[elem_idx] <= {flash_data, float_accum[31:8]};
                            byte_in_float      <= 2'd0;
                            elem_idx           <= elem_idx + 1'b1;
                        end else begin
                            byte_in_float <= byte_in_float + 1'b1;
                        end
                    end
                    if (flash_done) begin
                        state <= S_STATS;
                    end
                end

                // TODO: mean_reg <= fp_div(sum_reg, DIM)
                //       var_reg  <= fp_sub(fp_div(sqsum_reg, DIM), fp_mul(mean_reg, mean_reg))
                //       stddev_reg <= fp_sqrt(fp_add(var_reg, EPS))
                // (EPS = 1e-5, igual que el C++)
                S_STATS: begin
                    idx     <= 0;
                    in_addr <= 0;
                    state   <= S_NORM;
                end

                // Segunda pasada: x[i] <= ((x[i]-mean)/stddev)*gamma[i] + beta[i]
                // TODO: conectar fp_sub/fp_div/fp_mul/fp_add reales; por ahora
                // solo se arma el flujo de control y el direccionamiento.
                S_NORM: begin
                    out_addr <= idx;
                    out_we   <= 1'b1;

                    if (idx == DIM - 1) begin
                        state <= S_DONE;
                    end else begin
                        idx     <= idx + 1'b1;
                        in_addr <= in_addr + 1'b1;
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