"""
pc_listener.py
--------------------------------------------------------------------------
Lado PC de la comunicacion UART con la Basys3.

Dos modos:

  --mode text     (default) Muestra crudo todo lo que llegue por el puerto
                   serie, tal cual. Sirve HOY MISMO para ver el mensaje de
                   "Basys3 SML: connection OK" y cualquier otro texto ASCII
                   que mande la placa (status, debug, etc.)

  --mode tokens    Interpreta el stream como IDs de token de 2 bytes
                   (big-endian, uint16), busca cada uno en vocab.txt, y
                   los imprime como texto legible -- exactamente el mismo
                   formato que aplica format_token() en el main.cpp
                   original (maneja los marcadores 'Ġ' -> espacio,
                   'Ċ' -> salto de linea, y oculta <|endoftext|>).

                   Este modo requiere que el modulo RTL de generacion de
                   tokens (pendiente de construir) mande cada token como
                   2 bytes big-endian por UART. Ese es el CONTRATO que
                   tiene que respetar el hardware para que este script
                   funcione: uint16 big-endian, uno por token generado.

Requiere: pip install pyserial

Uso:
  python pc_listener.py --list                        # listar puertos disponibles
  python pc_listener.py --port COM5                    # modo texto (banner de conexion)
  python pc_listener.py --port COM5 --mode tokens --vocab vocab.txt
"""

import argparse
import sys
import time

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("Falta pyserial. Instalalo con: pip install pyserial", file=sys.stderr)
    sys.exit(1)


def list_ports():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No se encontro ningun puerto serie conectado.")
        return
    print("Puertos disponibles:")
    for p in ports:
        print(f"  {p.device}  -  {p.description}")


def load_vocab(path: str):
    vocab = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vocab.append(line.rstrip("\n").rstrip("\r"))
    return vocab


def format_token(token_str: str) -> str:
    """Misma logica que format_token() en el main.cpp de referencia."""
    out = token_str.replace("\u0120", " ").replace("\u010A", "\n")
    if out == "<|endoftext|>":
        return ""
    return out


def run_text_mode(ser: serial.Serial):
    print("[modo texto] Esperando datos de la Basys3... (Ctrl+C para salir)\n")
    try:
        while True:
            data = ser.read(ser.in_waiting or 1)
            if data:
                sys.stdout.write(data.decode("utf-8", errors="replace"))
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[cerrado por el usuario]")


def run_token_mode(ser: serial.Serial, vocab_path: str):
    vocab = load_vocab(vocab_path)
    print(f"[modo tokens] Vocabulario cargado: {len(vocab)} entradas")
    print("[modo tokens] Esperando IDs de token de la Basys3... (Ctrl+C para salir)\n")

    buf = bytearray()
    try:
        while True:
            data = ser.read(ser.in_waiting or 1)
            if not data:
                continue
            buf.extend(data)

            # Consumir de a 2 bytes (uint16 big-endian) mientras haya suficientes
            while len(buf) >= 2:
                token_id = (buf[0] << 8) | buf[1]
                del buf[0:2]

                if token_id >= len(vocab):
                    print(f"\n[ADVERTENCIA] token_id={token_id} fuera de rango "
                          f"de vocab (max={len(vocab)-1}), se ignora", file=sys.stderr)
                    continue

                text = format_token(vocab[token_id])
                sys.stdout.write(text)
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n[cerrado por el usuario]")


def main():
    parser = argparse.ArgumentParser(description="Cliente PC para la Basys3 SML")
    parser.add_argument("--list", action="store_true", help="listar puertos serie disponibles y salir")
    parser.add_argument("--port", type=str, help="puerto serie, ej. COM5 o /dev/tty.usbserial-XXXX")
    parser.add_argument("--baud", type=int, default=115200, help="baud rate (default 115200, debe matchear uart_tx.v)")
    parser.add_argument("--mode", type=str, choices=["text", "tokens"], default="text")
    parser.add_argument("--vocab", type=str, default="vocab.txt", help="ruta a vocab.txt (solo modo tokens)")
    args = parser.parse_args()

    if args.list:
        list_ports()
        return

    if not args.port:
        print("Falta --port. Usa --list para ver los puertos disponibles.", file=sys.stderr)
        sys.exit(1)

    print(f"Conectando a {args.port} @ {args.baud} baud (8N1)...")
    try:
        ser = serial.Serial(
            port=args.port,
            baudrate=args.baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
        )
    except serial.SerialException as e:
        print(f"No se pudo abrir el puerto: {e}", file=sys.stderr)
        sys.exit(1)

    # Pequeña pausa: algunos adaptadores USB-serie resetean la conexion al
    # abrirse, dar tiempo a que se estabilice antes de leer.
    time.sleep(0.2)
    ser.reset_input_buffer()

    try:
        if args.mode == "text":
            run_text_mode(ser)
        else:
            run_token_mode(ser, args.vocab)
    finally:
        ser.close()


if __name__ == "__main__":
    main()