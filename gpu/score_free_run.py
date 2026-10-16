"""Score free-twin runs (run.npz from dds_live_recorder): release, pad-load onset, fall time, time braced."""
import sys, os, glob, numpy as np
def score(d):
    f = os.path.join(d, "run.npz")
    if not os.path.exists(f): return None
    tr = np.load(f, allow_pickle=True); t = tr["t"]; z = tr["qpos_full"][:, 2]; q = tr["base_quat"]
    w, x, y, zz = q.T; r22 = 1 - 2 * (x * x + y * y); tilt = np.degrees(np.arccos(np.clip(r22, -1, 1)))
    names = [str(n) for n in tr["contact_names"]]; cf = tr["contact_force"]
    pad = [i for i, n in enumerate(names) if "pad" in n or "forearm" in n]
    padN = np.abs(cf[:, pad, 0]).sum(1) if pad else np.zeros(len(t))
    fall = next((t[i] for i in range(len(t)) if z[i] < 0.5), None)
    first_pad = next((t[i] for i in range(len(t)) if padN[i] > 30), None)
    braced = float(np.sum(np.diff(t)[(padN[:-1] > 30) & (z[:-1] > 0.5)])) if len(t) > 1 else 0
    lean = next((t[i] for i in range(len(t)) if tilt[i] > 10), None)
    # seat = pad > 100 N held for 1 s; ejection = pad drops below 10 N for > 0.3 s after first load, robot still up
    seat = None; ej = 0; up = z > 0.5
    if first_pad is not None:
        i0 = int(np.searchsorted(t, first_pad)); hot = padN > 100
        for i in range(i0, len(t)):
            if hot[i] and up[i] and t[min(len(t)-1, int(np.searchsorted(t, t[i] + 1.0)))] - t[i] >= 0.9 and hot[i:int(np.searchsorted(t, t[i] + 1.0))].all(): seat = t[i]; break
        loaded = False; off_t = None
        for i in range(i0, len(t)):
            if not up[i]: break
            if padN[i] > 30: loaded = True; off_t = None
            elif loaded and padN[i] < 10:
                if off_t is None: off_t = t[i]
                elif t[i] - off_t > 0.3: ej += 1; loaded = False; off_t = None
    return dict(dur=float(t[-1]), fall=fall, lean=lean, first_pad=first_pad, braced_s=braced, padN_max=float(padN.max()),
                tilt_max_pre_fall=float(tilt[z > 0.5].max()) if (z > 0.5).any() else float("nan"), names=names,
                seat=seat, load_to_seat=(seat - first_pad) if (seat is not None and first_pad is not None) else None, ejections=ej)
if __name__ == "__main__":
    for pat in sys.argv[1:]:
        for d in sorted(glob.glob(pat)):
            s = score(d)
            if s: print(f"{os.path.basename(d):55s} dur {s['dur']:6.1f} lean@{s['lean']} pad>30N@{s['first_pad']} seat@{s['seat']} load->seat {s['load_to_seat']} eject {s['ejections']} braced {s['braced_s']:5.1f}s padNmax {s['padN_max']:5.0f} tiltmax {s['tilt_max_pre_fall']:5.1f} fall@{s['fall']}")
    print("contact names:", s["names"] if s else None)
