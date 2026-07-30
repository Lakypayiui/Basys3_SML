// =============================================================================
// quant_linear.v  (v3 -- interfaz de rafaga contra flash_reader.v v2)
// -----------------------------------------------------------------------------
// Capa lineal cuantizada: equivalente en RTL de linear_int4() en el C++ de
// referencia. Esta version reemplaza el handshake req/valid byte-a-byte de
// v2 por el modo rafaga de flash_reader.v v2: se pide la fila ENTERA de
// pesos (IN_DIM/2 bytes) en un solo flash_start, y despues se consumen los
// bytes a medida que van llegando por flash_valid -- sin volver a pagar el
// header de comando+direccion en cada byte. Igual para scale (4 bytes) y
// bias (4 bytes).
//
// La aritmetica de punto flotante sigue marcada como TODO, sin cambios
// respecto a v2 -- esta revision es especificamente sobre el acceso a
// memoria (ya van dos rondas: v1->v2 fue "de ROM instantanea a Flash real",
// v2->v3 es "de Flash byte-a-byte a Flash en rafaga").
// =============================================================================

module quant_linear #(
    parameter IN_DIM     = 64,
    parameter OUT_DIM    = 64,
    parameter GROUP_SIZE = 1,        // filas por scale (ver group_size del .bin)
    parameter NUM_GROUPS = (OUT_DIM + GROUP_SIZE - 1) / GROUP_SIZE,
    parameter WEIGHT_BYTES_PER_ROW = (IN_DIM + 1) / 2
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,               // pulso: activaciones de entrada listas en in_mem
    output reg  done,                // pulso: salida lista en out_mem

    // Direcciones base en Flash para ESTA capa (las calcula quien orquesta
    // el forward pass, a partir del layout que escribio quantize.py)
    input  wire [23:0] weight_base,  // primer byte de datos INT4 empaquetados
    input  wire [23:0] scale_base,   // primer float32 de scale por grupo
    input  wire [23:0] bias_base,    // primer float32 de bias (si bias_en)
    input  wire        bias_en,

    // Memoria de activaciones de entrada (BRAM, rapida, 1 ciclo)
    output reg  [$clog2(IN_DIM)-1:0] in_addr,
    input  wire [31:0]               in_data,

    // Salida (activaciones resultado, BRAM, 1 ciclo)
    output reg  [$clog2(OUT_DIM)-1:0] out_addr,
    output reg  [31:0]                out_data,
    output reg                        out_we,

    // Bus compartido hacia flash_reader.v v2 (modo rafaga). Una sola
    // instancia de flash_reader en todo el diseño -- ver token_forward.v.
    output reg  [23:0] flash_addr,
    output reg  [15:0] flash_burst_len,
    output reg         flash_start,
    input  wire        flash_busy,
    input  wire [7:0]  flash_data,
    input  wire        flash_valid,
    input  wire        flash_done
);

    // -------------------------------------------------------------------
    // NOTA DE IMPLEMENTACION: la aritmetica en punto flotante (comparar
    // magnitudes, multiplicar por scales, sumar bias) sigue pendiente del
    // Floating-Point Operator IP de Vivado -- ver comentarios TODO abajo.
    // -------------------------------------------------------------------

    localparam S_IDLE        = 4'd0;
    localparam S_SCAN        = 4'd1;   // fase 1: max_abs de las activaciones (BRAM)
    localparam S_ROW_INIT    = 4'd2;   // decidir direcciones de la fila nueva
    localparam S_SCALE_START = 4'd3;   // lanzar rafaga de 4 bytes del scale de grupo
    localparam S_SCALE_WAIT  = 4'd4;   // consumir los 4 bytes
    localparam S_WEIGHT_START= 4'd5;   // lanzar rafaga de la fila de pesos completa
    localparam S_WEIGHT_WAIT = 4'd6;   // consumir bytes de a 1 (2 columnas c/u)
    localparam S_BIAS_START  = 4'd7;   // lanzar rafaga de 4 bytes del bias
    localparam S_BIAS_WAIT   = 4'd8;   // consumir los 4 bytes
    localparam S_WRITE_OUT   = 4'd9;   // escalar acumulador + bias, escribir salida
    localparam S_DONE        = 4'd10;

    reg [3:0] state;
    reg [$clog2(IN_DIM):0]   scan_idx;
    reg [$clog2(OUT_DIM):0]  row_idx;
    reg [$clog2(IN_DIM):0]   col_idx;

    reg [31:0] max_abs_reg;
    reg [7:0]  in_scale_shift;  // aproximacion de in_scale como shift (ver v1)

    reg signed [31:0] acc;      // acumulador entero (INT4 x INT8, igual que el C++)

    // Cache del scale de grupo (igual criterio que v2: solo se re-lee de
    // Flash cuando row_idx cruza a un grupo nuevo)
    reg [31:0] cached_scale;
    reg [$clog2(NUM_GROUPS)-1:0] cached_group;
    reg        have_cached_scale;

    reg [31:0] float_accum;  // ensamblado little-endian para scale/bias (4 bytes)
    reg [31:0] bias_reg;
    reg [7:0]  weight_byte_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state             <= S_IDLE;
            done              <= 1'b0;
            out_we            <= 1'b0;
            max_abs_reg       <= 32'd0;
            scan_idx          <= 0;
            row_idx           <= 0;
            col_idx           <= 0;
            acc               <= 32'sd0;
            have_cached_scale <= 1'b0;
            flash_start       <= 1'b0;
        end else begin
            done        <= 1'b0;
            out_we      <= 1'b0;
            flash_start <= 1'b0; // pulso por defecto en bajo

            case (state)
                S_IDLE: begin
                    if (start) begin
                        scan_idx          <= 0;
                        max_abs_reg       <= 32'd0;
                        in_addr           <= 0;
                        have_cached_scale <= 1'b0;
                        state             <= S_SCAN;
                    end
                end

                // Fase 1: sin cambios respecto a v1/v2 (BRAM, no toca Flash).
                // TODO: comparacion real de magnitud float.
                S_SCAN: begin
                    if (in_data[30:0] > max_abs_reg[30:0]) begin
                        max_abs_reg <= {1'b0, in_data[30:0]};
                    end
                    if (scan_idx == IN_DIM - 1) begin
                        in_scale_shift <= max_abs_reg[30:23]; // placeholder, ver v1
                        row_idx  <= 0;
                        state    <= S_ROW_INIT;
                    end else begin
                        scan_idx <= scan_idx + 1'b1;
                        in_addr  <= in_addr + 1'b1;
                    end
                end

                // Decide si hace falta un scale de grupo nuevo de Flash.
                S_ROW_INIT: begin
                    acc     <= 32'sd0;
                    col_idx <= 0;
                    in_addr <= 0;

                    if (!have_cached_scale || (row_idx / GROUP_SIZE) != cached_group) begin
                        cached_group    <= row_idx / GROUP_SIZE;
                        flash_addr      <= scale_base + ((row_idx / GROUP_SIZE) << 2);
                        flash_burst_len <= 16'd4;
                        flash_start     <= 1'b1;
                        state           <= S_SCALE_START;
                    end else begin
                        flash_addr      <= weight_base + row_idx * WEIGHT_BYTES_PER_ROW;
                        flash_burst_len <= WEIGHT_BYTES_PER_ROW;
                        flash_start     <= 1'b1;
                        state           <= S_WEIGHT_START;
                    end
                end

                // Ya se emitio flash_start; esperar a que flash_reader
                // confirme que arranco (busy=1) antes de pasar a consumir.
                S_SCALE_START: begin
                    if (flash_busy) state <= S_SCALE_WAIT;
                end

                S_SCALE_WAIT: begin
                    if (flash_valid) begin
                        float_accum <= {flash_data, float_accum[31:8]};
                    end
                    if (flash_done) begin
                        cached_scale      <= float_accum; // ya tiene los 4 bytes
                        have_cached_scale <= 1'b1;
                        flash_addr        <= weight_base + row_idx * WEIGHT_BYTES_PER_ROW;
                        flash_burst_len   <= WEIGHT_BYTES_PER_ROW;
                        flash_start       <= 1'b1;
                        state             <= S_WEIGHT_START;
                    end
                end

                S_WEIGHT_START: begin
                    if (flash_busy) state <= S_WEIGHT_WAIT;
                end

                // Cada flash_valid trae 1 byte = 2 columnas (nibble bajo =
                // par, alto = impar, igual empaquetado que pack_int4() en
                // quantize.py). flash_done indica que ya llegaron los
                // WEIGHT_BYTES_PER_ROW bytes de esta fila completa.
                S_WEIGHT_WAIT: begin
                    if (flash_valid) begin
                        weight_byte_reg <= flash_data;
                        // TODO: desempaquetar los 2 nibbles con sign-extension,
                        // re-cuantizar in_data a INT8 con in_scale_shift, y
                        // acumular en acc los productos de las columnas
                        // col_idx y col_idx+1 (ver detalle omitido en v1/v2).
                        col_idx <= col_idx + 2'd2;
                        in_addr <= in_addr + 2'd2;
                    end
                    if (flash_done) begin
                        if (bias_en) begin
                            flash_addr      <= bias_base + (row_idx << 2);
                            flash_burst_len <= 16'd4;
                            flash_start     <= 1'b1;
                            state           <= S_BIAS_START;
                        end else begin
                            bias_reg <= 32'd0;
                            state    <= S_WRITE_OUT;
                        end
                    end
                end

                S_BIAS_START: begin
                    if (flash_busy) state <= S_BIAS_WAIT;
                end

                S_BIAS_WAIT: begin
                    if (flash_valid) begin
                        float_accum <= {flash_data, float_accum[31:8]};
                    end
                    if (flash_done) begin
                        bias_reg <= float_accum;
                        state    <= S_WRITE_OUT;
                    end
                end

                // TODO: out_data <= acc * cached_scale * in_scale + bias_reg
                // (conectar a fp_mul / fp_add, el IP de Vivado).
                S_WRITE_OUT: begin
                    out_addr <= row_idx;
                    out_we   <= 1'b1;

                    if (row_idx == OUT_DIM - 1) begin
                        state <= S_DONE;
                    end else begin
                        row_idx <= row_idx + 1'b1;
                        state   <= S_ROW_INIT;
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