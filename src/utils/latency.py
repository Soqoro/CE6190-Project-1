from __future__ import annotations
import time, torch

def measure_latency(model, device, input_size=(1,3,512,512), warmup=30, iters=100):
    x = torch.randn(*input_size, device=device)
    model.eval(); model.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0  # ms
