import json
import socket
from prediction_utils import load_all, predict_record

HOST = "127.0.0.1"
PORT = 5050

def main():
    metadata, models = load_all()
    print(f"Network Intrusion Detection socket server listening on {HOST}:{PORT}")
    print("Send one JSON traffic record per line. Press Ctrl+C to stop.")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(5)

        while True:
            connection, address = server.accept()
            with connection:
                print(f"\nClient connected: {address}")
                buffer = b""
                while True:
                    chunk = connection.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        try:
                            record = json.loads(line.decode("utf-8"))
                            result = predict_record(record, metadata, models)
                            response = {"status": "success", "predictions": result}
                        except Exception as exc:
                            response = {"status": "error", "message": str(exc)}
                        connection.sendall((json.dumps(response) + "\n").encode("utf-8"))
                print("Client disconnected.")

if __name__ == "__main__":
    main()
