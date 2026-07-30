// =============================================================================
// sml_top.v
// -----------------------------------------------------------------------------
// Top-level que ata: top_fsm (el menu) + uart_string_sender (comunicacion con
// la PC). Al prender la placa (o tras reset), manda "Basys3 SML: connection OK"
// por UART antes de habilitar el menu -- abrir una terminal serie en la PC
// (screen /dev/tty.usbserial-XXXX 115200 en Mac, o PuTTY/minicom equivalente)
// para verlo.
//
// El pipeline de inferencia real (prefill/generate) todavia no esta conectado:
// prefill_done y gen_done estan atados a contadores de ciclos como placeholder,
// SOLO para poder simular la FSM completa de punta a punta antes de que el
// pipeline de computo (quant_linear + atencion + top-k) este terminado. Hay
// que reemplazar esos dos contadores por las señales reales de done del
// pipeline en cuanto ese pipeline exista.
// =============================================================================

module sml_top #(
    parameter CLK_HZ = 100_000_000,
    parameter BAUD   = 115200
)(
    input  wire clk,        // reloj de 100MHz del Basys3
    input  wire btnR,       // boton de reset fisico, ACTIVO EN ALTO (como todos los de Basys3)

    input  wire btnC,
    input  wire btnU,
    input  wire btnD,

    output wire [4:0] led,
    output wire        uart_tx_pin  // conectar al pin del bridge UART-USB (ver .xdc de Digilent)
);

    // -------------------------------------------------------------------
    // Antirrebote de btnR antes de derivar rst_n. Sin esto, el rebote
    // mecanico del boton (varios ms de 0/1 alternando al soltarlo) reinicia
    // la logica varias veces seguidas -- exactamente lo que corta la
    // transmision del mensaje de boot a mitad de camino y la reinicia.
    //
    // OJO: como esto mismo genera el reset del sistema, no puede resetearse
    // a si mismo con rst_n (careria en circulo). Arranca en el estado dado
    // por la inicializacion de FF tras la configuracion del FPGA (GSR).
    // -------------------------------------------------------------------
    reg btnR_sync0, btnR_sync1;
    reg btnR_stable;
    reg [19:0] db_cnt_r;

    always @(posedge clk) begin
        btnR_sync0 <= btnR;
        btnR_sync1 <= btnR_sync0;
    end

    always @(posedge clk) begin
        if (btnR_sync1 == btnR_stable) begin
            db_cnt_r <= 20'd0;
        end else begin
            db_cnt_r <= db_cnt_r + 1'b1;
            if (db_cnt_r == 20'hFFFFF) btnR_stable <= btnR_sync1; // ~10ms estable
        end
    end

    // btnR es activo en alto; rst_n interno es activo en bajo.
    wire rst_n = ~btnR_stable;

    // -------------------------------------------------------------------
    // Mensaje de boot: se pasa directo como parametro string a
    // uart_string_sender (ver MSG_DATA mas abajo). Verilog empaqueta el
    // string literal automaticamente en un vector de bits -- no hace falta
    // una ROM/array por separado (los puertos de tipo array no son
    // sintetizables en archivos .v puros de Vivado).
    // -------------------------------------------------------------------
    localparam MSG_LEN = 27; // "Basys3 SML: connection OK\r\n"

    // -------------------------------------------------------------------
    // Handshake FSM <-> UART
    // -------------------------------------------------------------------
    wire send_banner;
    wire banner_done;

    uart_string_sender #(
        .CLK_HZ  (CLK_HZ),
        .BAUD    (BAUD),
        .MSG_LEN (MSG_LEN),
        .MSG_DATA("Basys3 SML: connection OK\r\n")
    ) u_boot_banner (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (send_banner),
        .done    (banner_done),
        .tx_pin  (uart_tx_pin)
    );

    // -------------------------------------------------------------------
    // Placeholder del pipeline de inferencia (a reemplazar por el pipeline
    // real: prefill de 4 tokens + generacion autoregresiva con quant_linear,
    // atencion, top-k). Por ahora, simples contadores para poder simular
    // la FSM completa de punta a punta.
    // -------------------------------------------------------------------
    reg [23:0] prefill_cnt, gen_cnt;
    reg        prefill_done_r, gen_done_r;
    wire       start_prefill, start_generate, request_new_seed;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prefill_cnt    <= 0;
            gen_cnt        <= 0;
            prefill_done_r <= 1'b0;
            gen_done_r     <= 1'b0;
        end else begin
            prefill_done_r <= 1'b0;
            gen_done_r     <= 1'b0;

            if (start_prefill) prefill_cnt <= 24'd0;
            else if (prefill_cnt < 24'd999_999) prefill_cnt <= prefill_cnt + 1'b1;
            else prefill_done_r <= 1'b1; // PLACEHOLDER: reemplazar por el pipeline real

            if (start_generate) gen_cnt <= 24'd0;
            else if (gen_cnt < 24'd4_999_999) gen_cnt <= gen_cnt + 1'b1;
            else gen_done_r <= 1'b1; // PLACEHOLDER: reemplazar por el pipeline real
        end
    end

    top_fsm #(
        .CLK_HZ(CLK_HZ)
    ) u_top_fsm (
        .clk               (clk),
        .rst_n             (rst_n),
        .btnC_raw          (btnC),
        .btnU_raw          (btnU),
        .btnD_raw          (btnD),
        .start_prefill     (start_prefill),
        .prefill_done      (prefill_done_r),
        .start_generate    (start_generate),
        .gen_done          (gen_done_r),
        .request_new_seed  (request_new_seed),
        .send_banner       (send_banner),
        .banner_done       (banner_done),
        .led               (led)
    );

endmodule