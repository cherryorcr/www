"""Receive one public model file over the encrypted Tailnet, verifying its digest."""
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import threading

TARGET = Path("/home/pc/research/TAG_condensation/experiments/uq_final_evidence_20261012/models/qwen2.5-1.5b/model.safetensors")
SIZE = 3087467144
DIGEST = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def reply(self, status, value):
        encoded = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path != "/health":
            self.reply(404, {"error": "unknown path"})
            return
        self.reply(200, {"ready": True, "expected_bytes": SIZE})

    def do_PUT(self):
        if self.path != "/qwen-pinned" or int(self.headers.get("Content-Length", "0")) != SIZE:
            self.reply(400, {"error": "unexpected path or byte count"})
            self.close_connection = True
            return
        temporary = TARGET.with_suffix(".safetensors.tailnet.tmp")
        digest, received = hashlib.sha256(), 0
        self.connection.settimeout(90)
        with temporary.open("wb") as stream:
            while received < SIZE:
                block = self.rfile.read(min(4 * 1024**2, SIZE - received))
                if not block:
                    raise ConnectionError("incomplete model transfer")
                stream.write(block)
                digest.update(block)
                received += len(block)
        actual = digest.hexdigest()
        if actual != DIGEST:
            self.reply(422, {"error": "official digest mismatch", "received": received})
            return
        temporary.replace(TARGET)
        receipt = dict(bytes=received, sha256=actual, verified=True,
                       transfer="HTTP bound to encrypted Tailscale interface; canonical source and pinned digest")
        TARGET.with_name("transfer_receipt.json").write_text(json.dumps(receipt, indent=2))
        self.reply(200, receipt)
        threading.Thread(target=self.server.shutdown, daemon=True).start()


if __name__ == "__main__":
    # Bind only to the explicitly authorized research server's Tailnet address.
    HTTPServer(("100.126.175.112", 18765), Handler).serve_forever()
