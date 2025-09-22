import torch

def vram_snapshot(tag=""):
    if not torch.cuda.is_available():
        return {"tag": tag, "available": False}
    torch.cuda.synchronize()
    dev = torch.device('cuda:0')
    alloc = torch.cuda.memory_allocated(dev) / 1024**2
    reserv = torch.cuda.memory_reserved(dev) / 1024**2
    peak = torch.cuda.max_memory_allocated(dev) / 1024**2
    snap = {"tag": tag, "allocated_mb": alloc, "reserved_mb": reserv, "peak_mb": peak}
    print(f"[VRAM][{tag}] alloc={alloc:.0f}MB, reserved={reserv:.0f}MB, peak={peak:.0f}MB")
    return snap
