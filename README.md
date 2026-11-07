# DATA302 Final Project — Hand-Object Interaction Reconstruction from Monocular RGB Video

> **DRAFT** (2026-11-07). Items marked **[TBD]** will be fixed before release.

Design your own pipeline that takes a **monocular RGB video** (fixed camera, one right hand, one object)
and reconstructs

* the **MANO hand trajectory**,
* the **object mesh** (real size), and
* the **object 6D pose trajectory**,

all in the **camera frame** of the video.

**Deadline: 2026-12-07** [TBD: time, time zone]

---

## 1. Repository layout

Fork or copy this repository and keep this structure. Your code submission is this folder.

```
DATA302-HOI-Final-Project/
├── README.md                 # this file
├── docs/
│   └── SUBMISSION_FORMAT.md  # exact output conventions (READ THIS)
├── examples/
│   ├── example_01.json       # format example (dummy numbers)
│   └── example_01.glb        # format example (a box)
├── submodules/               # external models you use (git submodules)
├── mano_models/              # MANO_RIGHT.pkl goes here (NOT included, see §3)
├── main/                     # YOUR pipeline code
├── data/                     # the videos (download, see §4) — not committed
└── results/                  # your outputs: <video_id>.json + <video_id>.glb — not committed
```

## 2. Environment

Set up your own environment(s): the models you combine are your choice and have their own
requirements. Only the output format needs two things in addition:

* `smplx` to run the MANO forward pass, which needs the GitHub version of `chumpy` to read the
  official `.pkl`: `pip install smplx git+https://github.com/mattloper/chumpy`
  (`chumpy` 0.70 from PyPI fails to import with numpy ≥ 1.24).
* `trimesh` to write the `.glb` mesh.

## 3. MANO

The MANO model files may **not** be redistributed, so every student downloads them once:

1. Register at <https://mano.is.tue.mpg.de> and download **Models & Code**.
2. Copy `MANO_RIGHT.pkl` into `mano_models/` → `mano_models/MANO_RIGHT.pkl`.
3. Never commit it (`.gitignore` already excludes `*.pkl` in that folder).

The course uses **exactly** these settings (they change the meaning of the pose numbers):

```python
import smplx
mano = smplx.MANO("mano_models", is_rhand=True, use_pca=False, flat_hand_mean=True)
```

With `flat_hand_mean=False` the same 48 numbers give a different hand: on our example video
the fingertips move by 6 cm on average. See `docs/SUBMISSION_FORMAT.md` §3.

## 4. Data

Videos: **[TBD: download link]**. Put them under `data/`:

```
data/<video_id>/
├── frames/000000.jpg, 000001.jpg, …   # all F frames; file number = frame index t
└── meta.json                          # {"video_id", "level", "num_frames", "fps", "width", "height", "is_right"}
```

Video ids: `easy_01`, `easy_02`, `normal_01`, `normal_02`, `hard_01`, `hard_02`. All videos show a
**right hand**. **No camera intrinsics are given**: estimate them yourself (e.g. MoGe-3).

## 5. Your pipeline (`main/`, `submodules/`)

* Put your code in `main/` and the external models you use in `submodules/` (see the READMEs there).
* **One pipeline and one configuration for all videos**: no per-video code or parameters.
* Your code must regenerate your submitted results from `data/` (the TAs re-run a sample).

### Allowed / not allowed

* **Allowed**: manual point prompts for SAM3 (no need to automate target detection; SAM3's video
  propagation is useful). Manually choosing the best frame for SAM3D.
* **Not allowed**: any other manual intervention, e.g. manually adjusting the object scale; any
  ground-truth or annotation of the test videos from any source.
* **AI tools**: allowed for writing code only, not for producing masks, 3D estimates, etc.
  [TBD: wording aligned with the course slides]

## 6. Outputs and submission

For **every video** write two files into `results/` (format: `docs/SUBMISSION_FORMAT.md`):

```
results/
├── easy_01.json     # MANO params + MANO forward outputs + object pose + intrinsics
├── easy_01.glb      # final object mesh (scale applied, centred at its centroid)
├── easy_02.json
├── easy_02.glb
└── …
```

Then

```bash
cd results && zip ../results.zip *.json *.glb
```

| Where | What |
|---|---|
| **Gradescope** (leaderboard) | `results.zip` — any number of uploads [TBD: link] |
| **LMS** (final, by the deadline) | report (PDF), code (this folder, without `data/`, `mano_models/*.pkl`, checkpoints), and the **same** `results.zip` |

The leaderboard is feedback. The final grade is computed after the deadline from the LMS submission.

## 7. Grading (draft)

* Quantitative benchmark scores of your results, per video, weighted by difficulty:
  object trajectory error, hand trajectory error, contact error [TBD: final metric list].
  Before scoring, translations are aligned to the ground truth using the object **centroid**
  (`trimesh` centroid).
* Your scoring rule for object pose candidates.
* Design and analysis of your post-optimization.
* Report clarity and code quality.

## 8. Reference

* EasyHOI [TBD: link]
