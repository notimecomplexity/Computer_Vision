# PiRacer CNN Autopilot: RoboRacer @ Penn (CV Team)

This folder is the DonkeyCar application running on our **Waveshare PiRacer**. It holds the
car's driving code, our classical lane-keeping controller (**LaneKeeper**), and the
**CNN autopilots** trained on camera data from the car.

The goal of the CV team is to train neural networks that drive the car from camera input alone,
following the approach in *Pocket Racer: An accessible autonomous racing educational platform*
([Scientific Reports, 2026](https://www.nature.com/articles/s41598-026-49690-x)). That paper uses
**behavioral cloning**, which means training a model to copy a driver by mapping each camera frame
to steering. It compares CNNs, Vision Transformers (ViT), and CNN+LSTM models (LRCN).

Work so far by Darren (CV team).

---

## Summary

| Step | Status |
|---|---|
| Collected labelled driving data with LaneKeeper | ✅ 4,172 frames (2 sessions) |
| Trained a CNN on the Raspberry Pi (`pilot1`) | ✅ validation steering R² ≈ 0.94 |
| CNN drives the track autonomously (Local Pilot) | ✅ |
| Built a Docker pipeline to train on a MacBook and deploy to the Pi (`pilot2`) | ✅ |
| Compare against the paper's ViT / LRCN models | ⏳ next |

---

## Hardware and software

| | |
|---|---|
| Car | Waveshare PiRacer (Raspberry Pi, Pi Camera, PCA9685 PWM steering/throttle) |
| Car software | **DonkeyCar 4.2.1 (Waveshare fork)**, source at `~/projects/donkeycar` on the Pi |
| Python / TensorFlow on the Pi | Python 3.7.3, TensorFlow 1.13.1 (1.x series), virtualenv `~/env` |
| Training on Mac | Docker, Python 3.11, TensorFlow 2.15 (Apple Silicon, CPU) |

> The Pi runs an old TensorFlow 1.x. A Mac with Apple Silicon **cannot** run TF 1.13
> (it needs AVX CPU instructions that Rosetta does not provide), so models trained with modern
> TensorFlow have to be **converted on the Pi** (see [Training on a Mac](#option-b-train-on-a-mac-docker)).

---

## Folder layout

```
mycar/
├── manage.py             # Vehicle loop (team-customised): camera → preprocess → pilot → motors
├── config.py             # DonkeyCar defaults
├── myconfig.py           # Our overrides (camera, LaneKeeper, recording settings)
├── train.py              # DonkeyCar 4.2 training entry point
├── image_preprocess.py   # ImagePreprocessor part: rotate/undistort every camera frame
├── undistort.py          # Fisheye undistortion helper
├── calibrate.py          # Steering/throttle calibration
├── cv_parts/
│   └── lane_keeper.py    # LaneKeeper: classical CV lane-following controller
├── scripts/              # Misc helper scripts
├── convert_on_pi.py      # Turns Mac-trained weights (.npz) into a Pi-loadable .h5
├── models/               # Trained models (see below)
└── data/                 # Recorded driving data (NOT in git, see .gitignore)
```

---

## How driving works

Every camera frame goes through the same pipeline when recording and when driving:

```
PiCamera ─► cam/raw_image_array ─► ImagePreprocessor (rotate/undistort) ─► cam/image_array
                                                                         ├─► recorder (training data)
                                                                         └─► autopilot (CNN or LaneKeeper)
```

So the CNN sees the same processed images when driving as the ones it was trained on.
If the rotate/undistort settings change, retrain on newly recorded data.

### Which autopilot runs

`manage.py` selects the autopilot from the command line:

```python
use_cv_autopilot = getattr(cfg, 'USE_CV_AUTOPILOT', False) and not model_path
```

| Command | Autopilot in Local Pilot mode |
|---|---|
| `python manage.py drive` | **LaneKeeper** (`cv_parts/lane_keeper.py`) |
| `python manage.py drive --model models/<model>.h5` | **The CNN** (`KerasLinear`) |

To confirm, the startup log should show `Adding part KerasLinear` and no LaneKeeper part.

### Drive modes (web UI at `http://<PI_IP>:8887`)

| Mode | Steering | Throttle |
|---|---|---|
| User (`d`) | Human | Human |
| Local Angle | **Autopilot** | Human (joystick, or Throttle Mode = Constant) |
| Local Pilot | **Autopilot** | **Autopilot** |

Press **`d`** to return to User mode and stop.

---

## Data

Data is recorded in DonkeyCar's tub v2 format in `data/`:

```
data/
├── manifest.json            # metadata + deleted_indexes
├── catalog_N.catalog        # one JSON record per frame
└── images/<index>_cam_image_array_.jpg
```

Example record:

```json
{"_index": 0, "_session_id": "26-10-31_0", "cam/image_array": "0_cam_image_array_.jpg",
 "user/angle": -0.654, "user/mode": "local", "user/throttle": 0.4}
```

**The training set** was driven by **LaneKeeper**, so the CNN learns to imitate it:

| Session (Pi clock, UTC) | Frames | Use |
|---|---|---|
| `26-10-31_0` | 1,887 | Training |
| `26-11-07_1` | 2,285 | Training |
| `26-11-07_2` and later | small | CNN test drives, **excluded** (marked deleted / filtered by session) |

- Total training data: **4,172 frames** at 128×160 RGB
- Steering (`user/angle`, −1…+1): variance **0.0811**
- Throttle (`user/throttle`): **always 0.40**, so the CNN effectively learns a constant throttle

### Keeping CNN test drives out of the data

DonkeyCar records automatically while an autopilot drives. To stop that, `myconfig.py` sets:

```python
AUTO_RECORD_ON_THROTTLE = False
RECORD_DURING_AI = False
```

If you see `recorded … records` while the CNN drives, stop and check these settings.

---

## The CNN

We use DonkeyCar's `linear` model (`KerasLinear`). It's a 5-layer CNN very close to the 2D CNN in the
Pocket Racer paper:

| Layer | Output | Params |
|---|---|---|
| Input (`img_in`) | 128×160×3 | 0 |
| Conv 24 @5×5, stride 2 + Dropout | 62×78×24 | 1,824 |
| Conv 32 @5×5, stride 2 + Dropout | 29×37×32 | 19,232 |
| Conv 64 @5×5, stride 2 + Dropout | 13×17×64 | 51,264 |
| Conv 64 @3×3, stride 1 + Dropout | 11×15×64 | 36,928 |
| Conv 64 @3×3, stride 1 + Dropout | 9×13×64 | 36,928 |
| Flatten → Dense 100 → Dense 50 (+ Dropout) | | 753,950 |
| `n_outputs0` (steering), `n_outputs1` (throttle) | 1 + 1 | 102 |
| **Total** | | **900,228** |

Training settings: MSE loss on both outputs, Adam optimizer, batch size 128, 80/20 random
train/validation split, early stopping (patience 5, min delta 0.0005), best `val_loss` checkpoint kept.

---

## Training

### Option A: train on the Raspberry Pi

No version matching is needed, but it's slow (~3 min per epoch, ~45 min total).

```bash
ssh pi@<PI_IP>
source ~/env/bin/activate
cd ~/mycar
nohup python train.py --tubs ./data --model ./models/pilot1.h5 > train.log 2>&1 &
tail -f train.log          # Ctrl+C stops watching; training keeps running
```

**Fixes we needed on the Pi (one-time):**

1. `python manage.py train` is disabled in this version. Use `python train.py --tubs ...` (not `--tub`).
2. Missing `imgaug`. Install without letting pip build OpenCV from source:
   ```bash
   pip install --no-deps imgaug==0.4.0
   pip install --only-binary=:all: imageio scipy scikit-image Shapely
   ```
3. TF 1.x rejects `batch_size` when training from a dataset. We removed the line
   `batch_size=batch_size,` from `model.fit(...)` in
   `~/projects/donkeycar/donkeycar/parts/keras.py` (backup: `keras.py.bak`).
   **This patch is in the DonkeyCar source, not in this repo.**

Harmless messages: `Gdk-CRITICAL` / `Unable to init server` (no screen for plotting), and
`Object of type float32 is not JSON serializable` (a bug when writing `models/database.json` *after*
the model is saved; the end of that file gets cut off).

### Option B: train on a Mac (Docker)

This is much faster. It trains the **identical** architecture with TF 2.15, then converts on the Pi.
The files live in `~/roboracer` on the Mac (`Dockerfile`, `train_mac.py`, `convert_on_pi.py`).

```bash
# Mac: one-time
cd ~/roboracer && docker build -t roboracer-train .

# Mac: get data and train
rsync -av --progress pi@<PI_IP>:~/mycar/data/ ~/roboracer/data/
docker run --rm -it -v "$PWD":/work roboracer-train \
  python train_mac.py --tub data --out models/pilot2 --keep-sessions 26-10-31_0 26-11-07_1

# Mac: send weights to the car
scp models/pilot2.npz models/pilot2_check.json pi@<PI_IP>:~/mycar/models/

# Pi: convert into a TF1 .h5 (checks Mac and Pi predictions match)
cd ~/mycar && python convert_on_pi.py models/pilot2
```

`train_mac.py` options: `--split time` validates on the end of the recording (more honest than
random), and `--keep-sessions` keeps CNN test drives out of training.

---

## Results

| Model | Trained on | Data | Steering val MSE | Steering R²* | Track |
|---|---|---|---|---|---|
| `mypilot` | Pi, 2026-10-31 | earlier/smaller set | 0.052 | ≈ 0.36 | not used |
| **`pilot1`** | Pi, 2026-11-07 | 4,172 frames | **0.0046** | **≈ 0.94** | ✅ drives the track in Local Pilot |
| `pilot2` | Mac (Docker) → Pi | 4,172 frames | _TBD_ | _TBD_ | drives the track; slows on straights and speeds up in corners (under investigation) |

\*R² = 1 − (steering val MSE ÷ steering variance 0.0811). It uses DonkeyCar's random split, so
neighbouring near-identical frames land in both train and validation. That makes it **optimistic**.
Track behaviour is the real test.

`pilot1` training curve: val loss 0.0958 → 0.0047 over 15 epochs (stopped by early stopping).
The loss plot is `models/pilot1.png`.

---

## Running a model

```bash
ssh pi@<PI_IP>
source ~/env/bin/activate
cd ~/mycar
python manage.py drive --model ~/mycar/models/pilot1.h5
```

1. Lift the car's wheels off the ground and open `http://<PI_IP>:8887`.
2. Choose **Local Angle** and check that the front wheels steer with the camera view.
3. On the track, use Local Angle (Throttle Mode = Constant, 40%), then **Local Pilot**.
4. Press **`d`** to stop.

---

## Known issues

- **`pilot2` speed behaviour.** All training throttle labels are 0.40, so the model shouldn't vary
  speed. Next checks: compare against Local Angle with constant throttle, inspect the predicted
  throttle on the Mac, and look for throttle-modifying code in `manage.py` (`ThrottleFilter`,
  `AiLaunch`, `AI_THROTTLE_MULT`).
- `donkey tubplot` fails on the Pi (system pandas doesn't work with numpy 1.21).
- `donkey tubclean ./data` shows nothing. Use `donkey tubclean ~/mycar` and select `data`.
- The `mypilot` logs in `models/database.json` include one run with near-zero loss. That suggests
  near-constant steering labels in that early data.

---

## Next steps

1. Collect more LaneKeeper laps (aim for 10k+ frames), especially in corners and different lighting.
2. Retrain with `--split time` for an honest validation score.
3. Reproduce the paper's comparison: **2D CNN vs ViT vs LRCN** on the same data, comparing
   R², track completion and inference speed on the Pi.
4. Investigate a steering-only model with fixed throttle, as in the paper.

## Reference

Pocket Racer: An accessible autonomous racing educational platform. *Scientific Reports* (2026).
https://www.nature.com/articles/s41598-026-49690-x
