#!/usr/bin/env python3
"""
Run ON THE PI (in ~/mycar, with the (env) virtualenv active).

Builds DonkeyCar's own KerasLinear model with the Pi's TensorFlow 1.x, loads the
weights trained on the Mac (.npz), checks the predictions match the Mac's, and
saves a .h5 file that `python manage.py drive --model ...` can load.

Usage:
    python convert_on_pi.py models/pilot2
      -> reads  models/pilot2.npz and models/pilot2_check.json
      -> writes models/pilot2.h5
"""
import json
import os
import sys

import numpy as np
from PIL import Image

import donkeycar as dk
from donkeycar.utils import get_model_by_type

prefix = sys.argv[1] if len(sys.argv) > 1 else "models/pilot2"

cfg = dk.load_config(myconfig="myconfig.py")
kl = get_model_by_type("linear", cfg)
model = kl.model

# ---- load weights by layer name ----
data = np.load(prefix + ".npz")
loaded = 0
for layer in model.layers:
    n = len(layer.get_weights())
    if n == 0:
        continue
    keys = [f"{layer.name}/{i}" for i in range(n)]
    missing = [k for k in keys if k not in data]
    if missing:
        sys.exit(f"ERROR: Mac weights have no entry for layer '{layer.name}' ({missing}).")
    new = [data[k] for k in keys]
    for cur, w in zip(layer.get_weights(), new):
        if cur.shape != w.shape:
            sys.exit(f"ERROR: shape mismatch in '{layer.name}': Pi {cur.shape} vs Mac {w.shape}")
    layer.set_weights(new)
    loaded += 1
print(f"Loaded weights into {loaded} layers.")

# ---- verify against the Mac's predictions ----
check_path = prefix + "_check.json"
if os.path.exists(check_path):
    check = json.load(open(check_path))
    worst = 0.0
    for p in check["predictions"]:
        img = np.array(Image.open(os.path.join("data", "images", p["image"]))).astype(np.uint8)
        # 1) the raw network, with the same /255 scaling used on the Mac
        a, t = model.predict(img[None].astype(np.float32) / 255.0)
        # 2) DonkeyCar's own driving path (includes its own normalisation)
        try:
            ra, rt = kl.run(img)
        except Exception as e:  # older/newer signatures
            ra, rt = float("nan"), float("nan")
            print("  (kl.run not usable for the check:", e, ")")
        diff = abs(float(a[0][0]) - p["angle"])
        worst = max(worst, diff)
        print(f"  {p['image']}: mac {p['angle']:+.4f} | pi {float(a[0][0]):+.4f} "
              f"| donkey run {float(ra):+.4f}")
    print("max steering difference Mac vs Pi:", round(worst, 5))
    if worst > 0.01:
        print("WARNING: predictions differ by more than 0.01 -- do not drive with this model yet.")
    else:
        print("OK: Pi model matches the Mac model.")
    print("If 'donkey run' differs a lot from 'pi', DonkeyCar scales images differently "
          "when driving -- tell the person helping you before driving.")

out = prefix + ".h5"
model.save(out)
print("Saved", out, "-> drive with: python manage.py drive --model", out)
