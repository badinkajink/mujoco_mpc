"""Export a trained residual MLP (+ standardisation + clip) to the flat binary the C++ planner hook loads.
Layout (little-endian): int32 magic 0x4d4c5031 ('MLP1'), int32 nin, int32 nout, int32 nlayers; per layer int32 rows, cols,
float32 W[rows*cols] row-major, float32 b[rows]; then float32 mu[nin], sd[nin], ymu[nout], ysd[nout], clip[nout].
Activation SiLU between layers, none at the output."""
import sys, os, struct, numpy as np, torch
from common import *
tag = sys.argv[1] if len(sys.argv) > 1 else "dad"
st = torch.load(os.path.join(OUT, f"residual_mlp_{tag}.pt"), map_location="cpu"); nz = np.load(os.path.join(OUT, f"residual_norm_{tag}.npz"))
ids = sorted({int(k.split(".")[1]) for k in st if k.endswith(".weight")}); layers = [(st[f"net.{i}.weight"].numpy().astype(np.float32), st[f"net.{i}.bias"].numpy().astype(np.float32)) for i in ids]
nin, nout = layers[0][0].shape[1], layers[-1][0].shape[0]
outp = os.path.join(OUT, f"residual_{tag}" + (f"_clip{os.environ['MBRL_CLIP_SCALE']}" if os.environ.get("MBRL_CLIP_SCALE") else "") + ".bin")
with open(outp, "wb") as f:
    f.write(struct.pack("<iiii", 0x4d4c5031, nin, nout, len(layers)))
    for W, b in layers: f.write(struct.pack("<ii", *W.shape)); f.write(W.tobytes()); f.write(b.tobytes())
    cs = float(os.environ.get("MBRL_CLIP_SCALE", "1.0")); suffix = ""
    for k in ("mu", "sd", "ymu", "ysd", "clip"): f.write((np.asarray(nz[k], np.float32) * (cs if k == "clip" else 1.0)).tobytes())
print("wrote", outp, nin, nout, [W.shape for W, _ in layers])
