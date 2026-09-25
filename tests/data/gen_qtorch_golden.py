# Regenerates qtorch_golden.json (needs qtorch_plus==0.2.0): python tests/data/gen_qtorch_golden.py
import json, torch
from qtorch_plus.quant import posit_quantize, float_quantize
g = torch.Generator().manual_seed(1234)
x = torch.cat([
    torch.randn(1500, generator=g) * torch.exp2(torch.randint(-12, 12, (1500,), generator=g).float()),
    torch.tensor([0.0, -0.0, 1.0, -1.0, 2.0**-30, -(2.0**-30), 3e38, -3e38]),
])
out = {"inputs_hex": [f"{v:08x}" for v in (x.view(torch.int32).tolist())], "posit": {}, "float": {}}
for n in (4, 5, 6, 8):
    for es in (0, 1, 2):
        out["posit"][f"{n}_{es}"] = [f"{v & 0xffffffff:08x}" for v in posit_quantize(x, nsize=n, es=es).view(torch.int32).tolist()]
for e, m in ((4, 3), (5, 2)):
    out["float"][f"{e}_{m}"] = [f"{v & 0xffffffff:08x}" for v in float_quantize(x, exp=e, man=m, rounding="nearest").view(torch.int32).tolist()]
json.dump(out, open(__file__.replace("gen_qtorch_golden.py", "qtorch_golden.json"), "w"))
print(len(x), "inputs")
