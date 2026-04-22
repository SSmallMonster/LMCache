# SPDX-License-Identifier: Apache-2.0
"""Smoke test for DPUAgent.retrieve_kv_cache.

Run this manually on a host with CUDA, the DPU service running, and
libdpu_cache.so built:

    python -m lmcache.v1.storage_backend.dpu.smoke_retrieve_kv_cache \
        --dpu-ip 192.168.100.2 \
        --host-pci-addr 0000:ba:00.0 \
        --gpu-id 0
"""

# Standard
import argparse
import sys
import time

# Third Party
import torch

# First Party
from dpu import DPUAgent, DPUConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the smoke test.

    Returns:
        Parsed arguments containing the DPU connection settings and tensor shape.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpu-ip", default="192.168.100.2")
    parser.add_argument("--host-pci-addr", default="0000:ba:00.0")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-concurrent-ops", type=int, default=4)
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[2, 4],
        help="Tensor shape for both K and V. Must have at most 4 dimensions.",
    )
    return parser.parse_args()


def main() -> int:
    """Store a small KV pair to DPU, retrieve it, and verify equality.

    Returns:
        Process exit code. Zero indicates the retrieved tensors match.
    """
    args = parse_args()
    if len(args.shape) > 4:
        raise ValueError("DPU cache header supports at most 4 tensor dimensions")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this DPU smoke test")

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    key_id = f"retrieve_kv_cache_smoke_{time.time_ns()}"
    numel = 1
    for dim in args.shape:
        numel *= dim

    k_tensor = torch.arange(numel, dtype=torch.float16, device=device).reshape(args.shape)
    v_tensor = (k_tensor + 100).contiguous()

    config = DPUConfig(
        dpu_ip=args.dpu_ip,
        host_pci_addr=args.host_pci_addr,
        gpu_id=args.gpu_id,
        max_concurrent_ops=args.max_concurrent_ops,
    )
    agent = DPUAgent(config)

    print(f"storing key={key_id}")
    if not agent.store_kv_cache(key_id, k_tensor, v_tensor):
        raise RuntimeError("store_kv_cache returned False")

    print(f"retrieving key={key_id}")
    result = agent.retrieve_kv_cache(key_id, target_device=str(device))
    if result is None:
        raise RuntimeError("retrieve_kv_cache returned None")

    retrieved_k, retrieved_v = result
    torch.cuda.synchronize(device)

    if not torch.equal(retrieved_k, k_tensor):
        raise AssertionError("retrieved K tensor does not match stored K tensor")
    if not torch.equal(retrieved_v, v_tensor):
        raise AssertionError("retrieved V tensor does not match stored V tensor")

    print("retrieve_kv_cache smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
