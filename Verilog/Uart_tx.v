// =============================================================================
// uart_tx.v
// -----------------------------------------------------------------------------
// Transmisor UART simple (8N1: 8 bits de datos, sin paridad, 1 bit de stop).
// Pensado para conectarse al bridge USB-UART on-board del Basys3 (el mismo
// puerto que usa Vivado/programador para JTAG+UART, aparece como un puerto
// serie virtual en la PC -- ej. /dev/tty.usbserial-XXXX en Mac, COMx en
// Windows). Abrir ese puerto en una terminal (screen, minicom, PuTTY, etc.)
// a la velocidad BAUD configurada abajo para ver el texto generado.
//
// IMPORTANTE: el pin fisico exacto del TX del FPGA hacia el bridge UART-USB
// (frecuentemente nombrado "uart_rxd_out" en el XDC oficial de Digilent,
// porque desde el punto de vista del chip FTDI es su entrada RX) hay que
// tomarlo del archivo .xdc oficial de Basys3 que provee Digilent -- no lo
// hardcodeo aca para no arriesgar un pinout incorrecto. En el .xdc, buscar
// la señal de UART y mapearla al puerto `tx_pin` de este modulo.
// =============================================================================

module uart_tx #(
    parameter CLK_HZ  = 100_000_000,
    parameter BAUD    = 115200
)(
    input  wire clk,
    input  wire rst_n,

    input  wire       tx_start,   // pulso de 1 ciclo: "mandar tx_data"
    input  wire [7:0] tx_data,
    output reg        tx_busy,    // alto mientras se esta transmitiendo el byte actual
    output reg        tx_pin      // salida serie, idle en alto (1)
);

    localparam integer CYCLES_PER_BIT = CLK_HZ / BAUD;

    localparam S_IDLE  = 2'd0;
    localparam S_START = 2'd1;
    localparam S_DATA  = 2'd2;
    localparam S_STOP  = 2'd3;

    reg [1:0]  state;
    reg [15:0] cycle_cnt;   // suficiente para CYCLES_PER_BIT a 100MHz/115200 (~868)
    reg [2:0]  bit_idx;
    reg [7:0]  shift_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            tx_pin    <= 1'b1;   // linea idle en alto
            tx_busy   <= 1'b0;
            cycle_cnt <= 16'd0;
            bit_idx   <= 3'd0;
            shift_reg <= 8'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    tx_pin <= 1'b1;
                    if (tx_start) begin
                        shift_reg <= tx_data;
                        tx_busy   <= 1'b1;
                        cycle_cnt <= 16'd0;
                        state     <= S_START;
                    end
                end

                S_START: begin
                    tx_pin <= 1'b0; // bit de start
                    if (cycle_cnt == CYCLES_PER_BIT - 1) begin
                        cycle_cnt <= 16'd0;
                        bit_idx   <= 3'd0;
                        state     <= S_DATA;
                    end else begin
                        cycle_cnt <= cycle_cnt + 1'b1;
                    end
                end

                S_DATA: begin
                    tx_pin <= shift_reg[0];
                    if (cycle_cnt == CYCLES_PER_BIT - 1) begin
                        cycle_cnt <= 16'd0;
                        shift_reg <= shift_reg >> 1;
                        if (bit_idx == 3'd7) begin
                            state <= S_STOP;
                        end else begin
                            bit_idx <= bit_idx + 1'b1;
                        end
                    end else begin
                        cycle_cnt <= cycle_cnt + 1'b1;
                    end
                end

                S_STOP: begin
                    tx_pin <= 1'b1; // bit de stop
                    if (cycle_cnt == CYCLES_PER_BIT - 1) begin
                        cycle_cnt <= 16'd0;
                        tx_busy   <= 1'b0;
                        state     <= S_IDLE;
                    end else begin
                        cycle_cnt <= cycle_cnt + 1'b1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule


// =============================================================================
// uart_string_sender.v
// -----------------------------------------------------------------------------
// Manda un string fijo, caracter por caracter, por uart_tx. Usado hoy para el
// mensaje de "connection OK" en el boot. El mismo patron (una ROM de bytes +
// esta FSM chica) es el que despues va a usar el pipeline de generacion para
// mandar cada token de texto a medida que se produce -- por eso conviene
// dejarlo como modulo generico y reusable, no atado solo al mensaje de boot.
// =============================================================================

module uart_string_sender #(
    parameter CLK_HZ  = 100_000_000,
    parameter BAUD    = 115200,
    parameter MSG_LEN = 32,
    // El mensaje entero empaquetado en un solo vector de bits (8*MSG_LEN bits).
    // Se pasa como un literal de string de Verilog en la instanciacion, ej:
    //   .MSG_DATA("Basys3 SML: connection OK\r\n")
    // Verilog empaqueta el string con el PRIMER caracter en los bits mas
    // significativos, por eso char_idx=0 lee el byte mas alto (ver logica
    // de indexado abajo). Evitamos asi un puerto de tipo array, que Vivado
    // no acepta en archivos .v puros (solo en SystemVerilog / .sv).
    parameter [8*MSG_LEN-1:0] MSG_DATA = {(8*MSG_LEN){1'b0}}
)(
    input  wire clk,
    input  wire rst_n,

    input  wire start,      // pulso: empezar a mandar el mensaje
    output reg  done,       // alto un ciclo cuando termino de mandar todo el mensaje
    output wire tx_pin      // conectar directo al pin fisico de UART
);

    localparam S_IDLE = 1'd0;
    localparam S_SEND = 1'd1;

    reg        state;
    reg [7:0]  char_idx;
    reg        tx_start;
    wire       tx_busy;

    // Byte actual: char_idx=0 -> byte mas significativo de MSG_DATA (primer
    // caracter del string), char_idx=MSG_LEN-1 -> byte menos significativo.
    wire [7:0] cur_char = MSG_DATA[8*(MSG_LEN-1-char_idx) +: 8];

    uart_tx #(
        .CLK_HZ(CLK_HZ),
        .BAUD(BAUD)
    ) u_uart_tx (
        .clk      (clk),
        .rst_n    (rst_n),
        .tx_start (tx_start),
        .tx_data  (cur_char),
        .tx_busy  (tx_busy),
        .tx_pin   (tx_pin)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            char_idx <= 8'd0;
            tx_start <= 1'b0;
            done     <= 1'b0;
        end else begin
            tx_start <= 1'b0;
            done     <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        char_idx <= 8'd0;
                        tx_start <= 1'b1; // manda el primer caracter
                        state    <= S_SEND;
                    end
                end

                S_SEND: begin
                    // Esperar a que uart_tx termine el caracter actual, despues
                    // avanzar al siguiente (o terminar si ya se mando todo).
                    if (!tx_busy && !tx_start) begin
                        if (char_idx == MSG_LEN - 1) begin
                            done  <= 1'b1;
                            state <= S_IDLE;
                        end else begin
                            char_idx <= char_idx + 1'b1;
                            tx_start <= 1'b1;
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule