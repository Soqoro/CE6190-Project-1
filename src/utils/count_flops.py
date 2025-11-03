from __future__ import annotations
try:
    from ptflops import get_model_complexity_info
except Exception:
    get_model_complexity_info = None

def count_flops_params(model, input_res=(3, 512, 512)):
    """Return (FLOPs, params). If ptflops is unavailable, returns (nan, param_count)."""
    import math
    params = sum(p.numel() for p in model.parameters())
    if get_model_complexity_info is None:
        return math.nan, params
    with get_model_complexity_info(model, input_res, as_strings=False, print_per_layer_stat=False) as (macs, params2):
        flops = macs * 2
        return flops, params2
