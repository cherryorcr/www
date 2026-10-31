"""Upload the two-payload candidate implementation to the GPU host."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/home/pc/research/TAG_condensation"
FILES = (
    "code/v2_llm_fusion/cat_multimodal_minimal.py",
    "code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py",
)


def main() -> None:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("100.126.175.112", username="pc",
                   password=os.environ["TAG_REMOTE_PASSWORD"], timeout=30)
    try:
        sftp = client.open_sftp()
        hashes = {}
        for name in FILES:
            local = ROOT / name
            remote = REMOTE_ROOT + "/" + name.replace("\\", "/")
            sftp.put(str(local), remote)
            hashes[name] = hashlib.sha256(local.read_bytes()).hexdigest()
        sftp.close()
        print(hashes)
    finally:
        client.close()


if __name__ == "__main__":
    main()
