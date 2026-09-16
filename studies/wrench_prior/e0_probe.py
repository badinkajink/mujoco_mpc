#!/usr/bin/env python3
"""E0 probe: ask a VLM for the physical-parameter prior on a photographed object.

Minimal prompt on purpose (the user's position on prompt engineering is on
record): one paragraph, a fixed JSON schema, point + 90% interval per quantity.
Run: set -a; source docker/.env; set +a; python3 e0_probe.py <image> "<object>" [n]
Writes the samples to stdout as JSON lines. The 2026-09-14 probe on the two
photographed objects from the 2025 experiments is runs/e0_gemini_probe.json.
"""
import json, os, sys, time
from PIL import Image
from google import genai

PROMPT = """You are the physics prior for a robot that will push or manipulate the object named below, shown in the left (overhead) view. Estimate the physical quantities the robot's planner needs. Give a point estimate and a 90% credible interval for each; be calibrated, not conservative.
Return ONLY JSON with keys: mass_kg [lo, point, hi]; mu_table [lo, point, hi] (sliding friction coefficient of the object on this table surface); f_tip_N [lo, point, hi] (lateral force at mid-height that tips it over); f_crush_N [lo, point, hi] (contact force that dents or crushes it); notes (one sentence).
Object: {obj}"""


def main():
    img, obj = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    model = os.environ.get("E0_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    im = Image.open(img)
    for k in range(n):
        r = client.models.generate_content(model=model, contents=[PROMPT.format(obj=obj), im])
        txt = r.text.strip().strip("`")
        txt = txt[txt.find("{"):txt.rfind("}") + 1]
        print(json.dumps({"model": model, "obj": obj, "sample": k, **json.loads(txt)}), flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
