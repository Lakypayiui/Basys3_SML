// =============================================================================
// flash_reader.v  (v2 -- modo rafaga)
// -----------------------------------------------------------------------------
// Lee bytes de la Flash QSPI del Basys3 (donde vive tinystories_1m_q4.bin,
// programado junto al bitstream via write_cfgmem) DURANTE la inferencia.
//
// Diferencia respecto a v1: en vez de repetir comando+direccion (32 bits)
// por CADA byte, ahora se manda el header UNA sola vez y despues se siguen
// clockeando bytes consecutivos con CS bajo (la Flash auto-incrementa la
// direccion internamente mientras el reloj sigue corriendo). Con esto, leer
// una fila de 64 columnas (32 bytes empaquetados) pasa de ~32 headers
// completos a 1 solo header + 32 bytes de datos -- varias veces mas rapido,
// y el prerequisito real para que la capa final (wte/lm_head, 50257 filas)
// tarde algo razonable.
//
// Sigue en SPI estandar (1 linea de datos), no Quad-SPI. Pasar a Quad
// (4 lineas DQ0-DQ3, comando 0x6B "Fast Read Quad Output") es la siguiente
// optimizacion de velocidad una vez que esto funcione en hardware real.
//
// =============================================================================

module flash_reader (
    input  wire        clk,        // 100MHz del sistema
    input  wire        rst_n,

    input  wire [23:0] addr,       // direccion de 24 bits donde arranca la rafaga
    input  wire [15:0] burst_len,  // cantidad de bytes a leer (1 = lectura simple)
    input  wire        start,      // pulso: arrancar la rafaga
    output reg         busy,       // alto mientras hay una rafaga en curso

    output reg  [7:0]  data_out,
    output reg         valid,      // pulso de 1 ciclo POR CADA byte entregado
    output reg         done,       // pulso de 1 ciclo cuando termino toda la rafaga

    // Pines fisicos hacia la flash (CCLK sale por STARTUPE2, no aca)
    output wire        flash_cs_n,
    output wire        flash_mosi,
    input  wire        flash_miso
);

    // -------------------------------------------------------------------
    // STARTUPE2: unica forma de manejar CCLK despues de la configuracion
    // en un 7-series.
    // -------------------------------------------------------------------
    reg cclk_reg;

    STARTUPE2 #(
        .PROG_USR("FALSE")
    ) u_startupe2 (
        .CFGCLK    (),
        .CFGMCLK   (),
        .EOS       (),
        .PREQ      (),
        .CLK       (1'b0),
        .GSR       (1'b0),
        .GTS       (1'b0),
        .KEYCLEARB (1'b0),
        .PACK      (1'b0),
        .USRCCLKO  (cclk_reg),
        .USRCCLKTS (1'b0),
        .USRDONEO  (1'b1),
        .USRDONETS (1'b1)
    );

    // -------------------------------------------------------------------
    // Divisor de reloj SPI (25MHz, conservador -- ver nota en v1 sobre
    // subir la velocidad como optimizacion posterior)
    // -------------------------------------------------------------------
    reg [1:0] clk_div;
    wire      spi_clk_en = (clk_div == 2'd3);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) clk_div <= 2'd0;
        else        clk_div <= clk_div + 1'b1;
    end

    // -------------------------------------------------------------------
    // FSM: CS bajo, mandar 0x03 + 24 bits de direccion UNA vez, despues
    // loopear burst_len bytes seguidos sin volver a mandar el header.
    // -------------------------------------------------------------------
    localparam S_IDLE = 3'd0;
    localparam S_HDR  = 3'd1;  // mandar comando (8b) + direccion (24b) = 32 bits, una vez
    localparam S_BYTE = 3'd2;  // leer un byte desde MISO
    localparam S_NEXT = 3'd3;  // decidir si sigue otro byte o termino la rafaga
    localparam S_DONE = 3'd4;

    reg [2:0]  state;
    reg [5:0]  bit_cnt;      // hasta 32 bits del header, o 8 del byte de datos
    reg [31:0] shift_out;    // comando + direccion
    reg [7:0]  shift_in;
    reg [15:0] bytes_done;
    reg        cs_n_reg;
    reg        mosi_reg;

    assign flash_cs_n = cs_n_reg;
    assign flash_mosi = mosi_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            cs_n_reg   <= 1'b1;
            cclk_reg   <= 1'b0;
            bit_cnt    <= 6'd0;
            bytes_done <= 16'd0;
            valid      <= 1'b0;
            done       <= 1'b0;
            busy       <= 1'b0;
            data_out   <= 8'd0;
        end else begin
            valid <= 1'b0;
            done  <= 1'b0;

            case (state)
                S_IDLE: begin
                    cclk_reg <= 1'b0;
                    if (start) begin
                        busy       <= 1'b1;
                        cs_n_reg   <= 1'b0;
                        shift_out  <= {8'h03, addr}; // comando Read + direccion 24b
                        bit_cnt    <= 6'd0;
                        bytes_done <= 16'd0;
                        state      <= S_HDR;
                    end
                end

                // Mandar los 32 bits del header (comando + direccion), una
                // sola vez por rafaga entera.
                S_HDR: begin
                    if (spi_clk_en) begin
                        if (cclk_reg == 1'b0) begin
                            mosi_reg <= shift_out[31];
                            cclk_reg <= 1'b1;
                        end else begin
                            shift_out <= {shift_out[30:0], 1'b0};
                            cclk_reg  <= 1'b0;
                            bit_cnt   <= bit_cnt + 1'b1;
                            if (bit_cnt == 6'd31) begin
                                bit_cnt <= 6'd0;
                                state   <= S_BYTE;
                            end
                        end
                    end
                end

                // Leer 8 bits de dato. A diferencia de v1, NO se vuelve a
                // mandar comando+direccion entre bytes -- la flash sigue
                // entregando bytes consecutivos mientras CS se mantenga
                // bajo y el reloj siga corriendo (auto-incremento interno).
                S_BYTE: begin
                    if (spi_clk_en) begin
                        if (cclk_reg == 1'b0) begin
                            cclk_reg <= 1'b1;
                        end else begin
                            shift_in <= {shift_in[6:0], flash_miso};
                            cclk_reg <= 1'b0;
                            bit_cnt  <= bit_cnt + 1'b1;
                            if (bit_cnt == 6'd7) begin
                                bit_cnt <= 6'd0;
                                state   <= S_NEXT;
                            end
                        end
                    end
                end

                S_NEXT: begin
                    data_out   <= shift_in;
                    valid      <= 1'b1;
                    bytes_done <= bytes_done + 1'b1;

                    if (bytes_done + 1'b1 == burst_len) begin
                        state <= S_DONE;
                    end else begin
                        state <= S_BYTE; // seguir con el proximo byte, sin re-mandar header
                    end
                end

                S_DONE: begin
                    cs_n_reg <= 1'b1;
                    busy     <= 1'b0;
                    done     <= 1'b1;
                    state    <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule