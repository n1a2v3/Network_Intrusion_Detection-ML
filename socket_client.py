import json
import socket
import sys

HOST = "127.0.0.1"
PORT = 5050

def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            record = json.load(f)
    else:
        # Replace/add values using the feature names from your dataset.
        record = {
            "duration": 0,
            "protocol_type": 1,
            "service": 49,
            "flag": 5,
            "src_bytes": 0,
            "dst_bytes": 0,
            "count": 120,
            "srv_count": 10,
            "serror_rate": 1.0,
            "srv_serror_rate": 1.0
        }

    with socket.create_connection((HOST, PORT), timeout=10) as client:
        client.sendall((json.dumps(record) + "\n").encode("utf-8"))
        response = client.makefile("r", encoding="utf-8").readline()
        print("\nSERVER RESPONSE")
        print(json.dumps(json.loads(response), indent=2))

if __name__ == "__main__":
    main()
