// =============================================================================
// top_fsm.v
// -----------------------------------------------------------------------------
// Maquina de estados de control (el "menu") para el SML corriendo en Basys3.
//
// Botones Basys3 usados (activos en alto, ya debounced/sincronizados abajo):
//   btnC  -> START / CONFIRM   (arrancar generacion / confirmar en menu)
//   btnU  -> REGENERAR         (pedir una historia nueva al terminar)
//   btnD  -> SALIR             (ir a estado de reposo)
//
// Salidas de estado (LEDs, para bring-up antes de tener UART/7-seg completo):
//   led[0] = IDLE
//   led[1] = PREFILL en curso
//   led[2] = GENERANDO
//   led[3] = DONE (historia lista, esperando input)
//   led[4] = EXIT (en reposo)
//
// Este modulo NO hace computo de la red neuronal. Expone un handshake simple
// (start_prefill / prefill_done / start_generate / gen_done) para conectarse
// con el pipeline de inferencia real (forward_token equivalente), que todavia
// no existe en RTL. Mientras ese pipeline no este listo, gen_done/prefill_done
// pueden atarse a un contador de ciclos para poder simular la FSM sola.
// =============================================================================

module top_fsm #(
    parameter CLK_HZ = 100_000_000
)(
    input  wire clk,
    input  wire rst_n,          // reset asincrono, activo en bajo (btnR tipicamente)

    // Botones crudos (sin debounce), tal cual vienen del pin fisico
    input  wire btnC_raw,
    input  wire btnU_raw,
    input  wire btnD_raw,

    // Handshake con el pipeline de inferencia (a implementar en modulos futuros)
    output reg  start_prefill,   // pulso de 1 ciclo: "arranca el prefill de los 4 tokens fijos"
    input  wire prefill_done,    // nivel alto cuando el prefill termino
    output reg  start_generate,  // pulso de 1 ciclo: "arranca la generacion autoregresiva"
    input  wire gen_done,        // nivel alto cuando se genero EOS o se llego a max_new_tokens
    output reg  request_new_seed,// pulso: pedir semilla nueva al LFSR para variar la historia

    // Handshake con el modulo que manda el mensaje de conexion por UART
    output reg  send_banner,     // pulso: "mandar el mensaje de conexion OK"
    input  wire banner_done,     // nivel alto cuando termino de mandarse

    // Estado visible para debug / bring-up
    output reg [4:0] led
);

    // -------------------------------------------------------------------
    // Debounce + deteccion de flanco de subida para cada boton.
    // A 100MHz, 20 bits de contador (~1M ciclos = 10ms) es un debounce
    // razonable para pulsadores mecanicos del Basys3.
    // -------------------------------------------------------------------
    // Sincronizador + contador de debounce por boton (~10ms a 100MHz)
    reg [19:0] db_cnt_c, db_cnt_u, db_cnt_d;
    reg btnC_stable, btnU_stable, btnD_stable;
    reg btnC_stable_d, btnU_stable_d, btnD_stable_d;
    wire btnC_pulse, btnU_pulse, btnD_pulse;

    // Sincronizador de 2 flip-flops para cruzar el dominio del boton fisico
    reg btnC_sync0, btnC_sync1;
    reg btnU_sync0, btnU_sync1;
    reg btnD_sync0, btnD_sync1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            btnC_sync0 <= 1'b0; btnC_sync1 <= 1'b0;
            btnU_sync0 <= 1'b0; btnU_sync1 <= 1'b0;
            btnD_sync0 <= 1'b0; btnD_sync1 <= 1'b0;
        end else begin
            btnC_sync0 <= btnC_raw; btnC_sync1 <= btnC_sync0;
            btnU_sync0 <= btnU_raw; btnU_sync1 <= btnU_sync0;
            btnD_sync0 <= btnD_raw; btnD_sync1 <= btnD_sync0;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            db_cnt_c <= 0; db_cnt_u <= 0; db_cnt_d <= 0;
            btnC_stable <= 1'b0; btnU_stable <= 1'b0; btnD_stable <= 1'b0;
        end else begin
            // Boton C
            if (btnC_sync1 == btnC_stable) db_cnt_c <= 0;
            else begin
                db_cnt_c <= db_cnt_c + 1'b1;
                if (db_cnt_c == 20'hFFFFF) btnC_stable <= btnC_sync1;
            end
            // Boton U
            if (btnU_sync1 == btnU_stable) db_cnt_u <= 0;
            else begin
                db_cnt_u <= db_cnt_u + 1'b1;
                if (db_cnt_u == 20'hFFFFF) btnU_stable <= btnU_sync1;
            end
            // Boton D
            if (btnD_sync1 == btnD_stable) db_cnt_d <= 0;
            else begin
                db_cnt_d <= db_cnt_d + 1'b1;
                if (db_cnt_d == 20'hFFFFF) btnD_stable <= btnD_sync1;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            btnC_stable_d <= 1'b0; btnU_stable_d <= 1'b0; btnD_stable_d <= 1'b0;
        end else begin
            btnC_stable_d <= btnC_stable;
            btnU_stable_d <= btnU_stable;
            btnD_stable_d <= btnD_stable;
        end
    end

    assign btnC_pulse = btnC_stable & ~btnC_stable_d;
    assign btnU_pulse = btnU_stable & ~btnU_stable_d;
    assign btnD_pulse = btnD_stable & ~btnD_stable_d;

    // -------------------------------------------------------------------
    // Estados de la maquina (el "menu")
    // -------------------------------------------------------------------
    localparam S_BOOT      = 3'd5;  // manda "connection OK" por UART antes de habilitar el menu
    localparam S_IDLE      = 3'd0;  // esperando btnC para arrancar la primera historia
    localparam S_PREFILL   = 3'd1;  // prefill de los 4 tokens fijos ("Once upon a time")
    localparam S_GENERATE  = 3'd2;  // generacion autoregresiva token a token
    localparam S_DONE      = 3'd3;  // historia terminada, esperando input del usuario
    localparam S_EXIT      = 3'd4;  // reposo, solo btnR (reset externo) saca de aca

    reg [2:0] state, next_state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S_BOOT;
        else        state <= next_state;
    end

    // -------------------------------------------------------------------
    // Logica de transicion
    // -------------------------------------------------------------------
    always @(*) begin
        next_state = state; // por defecto, quedarse
        case (state)
            S_BOOT: begin
                if (banner_done) next_state = S_IDLE;
            end

            S_IDLE: begin
                if (btnC_pulse) next_state = S_PREFILL;
            end

            S_PREFILL: begin
                if (prefill_done) next_state = S_GENERATE;
            end

            S_GENERATE: begin
                if (gen_done) next_state = S_DONE;
            end

            S_DONE: begin
                if (btnU_pulse)      next_state = S_PREFILL; // "otra historia"
                else if (btnD_pulse) next_state = S_EXIT;    // "salir"
            end

            S_EXIT: begin
                // Reposo total. Solo rst_n (boton de reset fisico del Basys3,
                // tipicamente btnR mapeado a rst_n externamente) vuelve a S_IDLE.
                next_state = S_EXIT;
            end

            default: next_state = S_IDLE;
        endcase
    end

    // -------------------------------------------------------------------
    // Salidas (pulsos de 1 ciclo hacia el pipeline de inferencia + LEDs de estado)
    // -------------------------------------------------------------------
    reg banner_sent;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            start_prefill    <= 1'b0;
            start_generate   <= 1'b0;
            request_new_seed <= 1'b0;
            send_banner      <= 1'b0;
            banner_sent      <= 1'b0;
            led              <= 5'b00000; // BOOT
        end else begin
            // Pulsos por defecto en bajo; se levantan solo en la transicion exacta
            start_prefill    <= 1'b0;
            start_generate   <= 1'b0;
            request_new_seed <= 1'b0;
            send_banner      <= 1'b0;

            case (state)
                S_BOOT: begin
                    led <= 5'b00000;
                    if (!banner_sent) begin
                        send_banner <= 1'b1; // pulso de 1 ciclo, una sola vez
                        banner_sent <= 1'b1;
                    end
                end

                S_IDLE: begin
                    led <= 5'b00001;
                    if (next_state == S_PREFILL) begin
                        start_prefill    <= 1'b1;
                        request_new_seed <= 1'b1; // primera semilla del LFSR
                    end
                end

                S_PREFILL: begin
                    led <= 5'b00010;
                    if (next_state == S_GENERATE) start_generate <= 1'b1;
                end

                S_GENERATE: begin
                    led <= 5'b00100;
                end

                S_DONE: begin
                    led <= 5'b01000;
                    if (next_state == S_PREFILL) begin
                        start_prefill    <= 1'b1;
                        request_new_seed <= 1'b1; // nueva historia -> nueva semilla
                    end
                end

                S_EXIT: begin
                    led <= 5'b10000;
                end

                default: led <= 5'b00001;
            endcase
        end
    end

endmodule