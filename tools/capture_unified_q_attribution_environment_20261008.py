"""Non-secret runtime metadata for the preregistered attribution matrix."""
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch


def main():
    output = Path(sys.argv[1])
    packages = {}
    for name in ("torch", "numpy", "scipy", "scikit-learn", "transformers", "tokenizers"):
        try:
            packages[name] = version(name)
        except Exception:
            packages[name] = None
    value = dict(python=sys.version, executable=sys.executable,
        platform=platform.platform(), packages=packages,
        torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        devices=[dict(name=torch.cuda.get_device_name(i),
                      total_memory_bytes=torch.cuda.get_device_properties(i).total_memory)
                 for i in range(torch.cuda.device_count())],
        gpu_driver=subprocess.check_output(["nvidia-smi", "--query-gpu=index,driver_version",
                                          "--format=csv,noheader"],text=True).strip(),
        worker_threads_declared=2,
        capture_process_defaults=dict(tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
        caveat="backend flags are defaults in this capture process; source hashes and runner record worker settings. CUDA construction replay is not asserted bitwise deterministic.")
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() and json.loads(output.read_text()) != value:
        raise ValueError("environment changed; preserve the previous record")
    output.write_text(json.dumps(value,indent=2),encoding="utf-8")
    print(json.dumps(value,indent=2))


if __name__ == "__main__":
    main()
