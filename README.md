<h1 align="center">Final Project : Hand-Object Interaction Reconstruction only from Monocular RGB Video (DATA302 2026)</h1>


> [!IMPORTANT]
> 1. **Read this README carefully YOURSELF before you start final project.** 
> 2. **Follow the output format ([§5](#5-final-output-format)) exactly.** A file that breaks the format is rejected. 

Design your own pipeline that takes a **monocular RGB video** (fixed camera) and reconstructs
* the **MANO hand trajectory**,
* the **object mesh** (real size), and
* the **object 6D pose trajectory**.

**Contents**

- [1. Repository layout](#1-repository-layout)
- [2. MANO](#2-mano)
- [3. Input videos](#3-input-videos)
- [4. Implementation](#4-your-pipeline)
- [5. Final output format](#5-final-output-format)
- [6. Leaderboard](#6-leaderboard)
- [7. Submission](#7-submission)

---

## 1. Repository layout

Fork or clone this repository to start your final project.

```
DATA302_Final_Project/
├── README.md          # 
├── examples/          # an example clip, a valid output for it, and its overlay (§5.3)
├── tools/overlay.py   # draws a result on its input frames -> overlay video (§5.3)
├── submodules/        # external models you use (git submodules)
├── mano_models/       # see §2
├── main/              # implement your pipeline here
├── data/              # see §3
└── results/           # your outputs: <video_id>.json + <video_id>.glb (§5) 
```

## 2. MANO

1. Register at <https://mano.is.tue.mpg.de> and download **Models & Code**.
2. Copy `MANO_RIGHT.pkl` into `mano_models/` → `mano_models/MANO_RIGHT.pkl`.
3. Never commit it (`.gitignore` already excludes `*.pkl` in that folder).

The course uses **exactly** these settings; they change the meaning of the pose numbers:

```python
import smplx
mano = smplx.MANO("mano_models", is_rhand=True, use_pca=False, flat_hand_mean=True)
```

## 3. Input videos

Download videos here: **<[LINK](https://drive.google.com/drive/folders/12O_mhWdEN5_P0Z08AX5L36V1zzegcjZ3?usp=drive_link)>**. Put them under `data/` and extract like below:

```
data/<video_id>/
└── frames/000000.jpg, 000001.jpg, …   # file number = frame index t
```

Video ids: `easy_01`, `easy_02`, `normal_01`, `normal_02`, `hard_01`, `hard_02`.

## 4. Your HOI reconstruction pipeline

* **Only** put your implemented pipeline code in `main/` (no debugging logs or visualizations).
* **One pipeline and one configuration for all videos**: no per-video code or parameters.
* Your code must regenerate your submitted results from `data/` (the TAs re-run a sample).

#### 🙆 Allowed

- **Manual prompts:** point prompts for SAM3; target detection does not need to be automated.
- **Frame selection:** manually choosing the best frame for SAM3D.
- **External models:** open-source, off-the-shelf models.

#### 🙅 Not allowed

- **Other manual intervention:** e.g. manually adjusting the object scale.
- **Cheating.**
- **External models:** commercial external models.

👾 **Frontier LLMs** (GPT, Claude, etc.) are allowed for coding only — not for generating masks, 3D object estimates, or any other results.


## 5. Final output format

> **Follow the output format below carefully when preparing your leaderboard submission.**

### Result files

Save **two files for every video** in `results/`:

1. A **JSON** file containing the hand and object trajectories. ([§5.1](#51-json-file-hand-and-object-trajectory))
2. A **GLB** file containing the object mesh. ([§5.2](#52-glb-file-object-mesh))

Use the **video ID as the filename** (e.g. `easy_01.json` and `easy_01.glb`).

```
results/
├── easy_01.json     # MANO parameters + MANO forward outputs + object pose + intrinsics
├── easy_01.glb      # final object mesh (real size, centroid at the origin)
├── easy_02.json
├── easy_02.glb
└── …                # 6 videos → 12 files
```

### Conventions overview

| Item | Convention |
|---|---|
| Camera frame | **OpenCV**: +x right, +y down, +z forward (into the scene). Express all trajectories in this fixed camera frame |
| Length unit | **meters** (vertices, joints, translations) |
| Intrinsics | **pixels**, at the resolution of the input frames |
| Frames | `F` is the total number of frames. Entry `t` corresponds to `frames/{t:06d}.jpg`. Provide results for **every frame**; interpolate missing estimates. All values must be finite (no NaN or Infinity) |
| Hand | right hand, `MANO_RIGHT.pkl`, `use_pca=False`, `flat_hand_mean=True` (§5.1) |
| Object rotation | 6D representation, **row** layout (§5.1) |
| Object mesh | `.glb`, scale applied (real size), **centroid at the origin** (§5.2) |
| Object pose | object → camera: `X_cam = R @ v + t` for every vertex `v` of the `.glb` |

### 5.1 JSON file (Hand and Object trajectory)

Save the final reconstructed hand and object trajectories and estimated intrinsics as **`<video_id>.json`**.

Specifically, each JSON file must contain:

- The **camera intrinsics**.
- **MANO parameters and forward outputs**.
- **Object poses** for all `F` frames.

The `object.mesh` field names the corresponding GLB file in the **same directory**.

**JSON structure**

The schema below includes comments and placeholders for clarity; omit them from your JSON.
See [`examples/results/example_01.json`](examples/results/example_01.json) for a complete example.

```text
{
  "format_version": "1.0",
  "video_id":   "easy_01",
  "num_frames": F,
  "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],  // [3,3] your estimate, pixels
  "mano": {
    "pose":     [[… 48 …], …],          // [F,48] axis-angle (radians): global_orient (3) + 15 finger joints (45)
    "transl":   [[x, y, z], …],         // [F,3]  meters, the `transl` argument of smplx.MANO
    "betas":    [… 10 …],               // [10]   one hand shape for the whole video
    "vertices": [[[x, y, z] × 778], …], // [F,778,3] smplx.MANO output.vertices, camera frame, meters
    "joints":   [[[x, y, z] × 16], …]   // [F,16,3]  smplx.MANO output.joints,   camera frame, meters
  },
  "object": {
    "mesh":   "easy_01.glb",            // file name of the mesh next to this JSON
    "rot6d":  [[… 6 …], …],             // [F,6] first two ROWS of R (object -> camera)
    "transl": [[x, y, z], …]            // [F,3] meters, position of the mesh origin = mesh centroid
  }
}
```

#### 5.1.1 Hand trajectory

Compute `vertices` and `joints` of MANO hand mesh from the `pose`, `betas`, and `transl` using the MANO
forward pass below.

The TAs will repeat this computation to verify that your parameters and outputs agree,
allowing for rounding when saving the JSON.

```python
import smplx, torch
mano = smplx.MANO("mano_models", is_rhand=True, use_pca=False, flat_hand_mean=True)
pose   = torch.as_tensor(pose,   dtype=torch.float32)          # (F, 48)
betas  = torch.as_tensor(betas,  dtype=torch.float32)          # (10,)
transl = torch.as_tensor(transl, dtype=torch.float32)          # (F, 3)
out = mano(global_orient=pose[:, :3], hand_pose=pose[:, 3:],
           betas=betas[None].expand(len(pose), -1), transl=transl)
vertices = out.vertices.detach().numpy()   # (F, 778, 3)
joints   = out.joints.detach().numpy()     # (F, 16, 3)
```

**Parameter conventions**

* **Pose convention**: `pose` contains 48 axis-angle values in radians: 3 for the global
  orientation and 45 for the 15 finger joints. Each finger rotation is relative to its parent
  joint. If your model returns rotation matrices of shape `(F, 16, 3, 3)`, convert them with
  `scipy.spatial.transform.Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(F, 48)`.

* **Hand mean**: use `flat_hand_mean=True`, so `hand_pose = 0` represents a flat, open hand.
  If your model uses `flat_hand_mean=False`, convert its pose parameters to this convention
  before running the forward pass.

* **Translation**: `transl` is the translation argument passed to `smplx.MANO`, in meters.
  It is applied after rotation about the wrist joint and is **not the wrist position**.

* **Hand shape**: use one `betas` vector for the entire video. If your model estimates a
  different shape for each frame, select or average those estimates and use the resulting
  vector for every frame.

#### 5.1.2 Object trajectory

**Rotation — `object.rot6d`**

`object.rot6d` stores the **first two rows** of the object-to-camera rotation matrix `R`,
following the convention used by PyTorch3D
`matrix_to_rotation_6d` / `rotation_6d_to_matrix`:

```python
import numpy as np
rot6d = R[:, :2, :].reshape(F, 6)                      # R: (F, 3, 3)

# back to R (this is what the grader does)
a1, a2 = rot6d[:, :3], rot6d[:, 3:]
b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
b2 = a2 - (b1 * a2).sum(1, keepdims=True) * b1
b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
R = np.stack([b1, b2, np.cross(b1, b2)], axis=1)        # rows b1, b2, b3
```

> **Do not use the first two columns**: decoding them as rows produces the transposed (inverse)
> rotation.

**Translation — `object.transl`**

`object.transl` stores the position of the **object mesh centroid** (§5.2.1) in the camera frame, in meters.

Together, the rotation and translation map each saved mesh vertex to the camera frame:

```text
X_cam = R @ v + t
```

If you center the mesh before export, update the translations as shown
in §5.2.1 **before writing the JSON**.

#### 5.1.3 Writing the JSON

Convert arrays to **plain JSON lists** and round floating-point values to **five decimal places**.

In this example:

- `K` is the camera intrinsic matrix.
- `rot6d` contains the encoded object rotations.
- `t` contains the translations for the **centered mesh** (§5.2.1).

```python
import json, numpy as np
r = lambda a: np.round(np.asarray(a, dtype=np.float64), 5).tolist()
sub = {
    "format_version": "1.0", "video_id": video_id, "num_frames": F,
    "intrinsics": r(K),
    "mano": {"pose": r(pose), "transl": r(transl), "betas": r(betas),
             "vertices": r(vertices), "joints": r(joints)},
    "object": {"mesh": f"{video_id}.glb", "rot6d": r(rot6d), "transl": r(t)},
}
with open(f"results/{video_id}.json", "w") as f:
    json.dump(sub, f, separators=(",", ":"), allow_nan=False)   # fails if anything is NaN
```

> `allow_nan=False` raises an error if any value is **NaN or Infinity**. Resolve missing or invalid
> estimates before saving.

### 5.2 GLB file (Object mesh)

Save the final reconstructed object as **`<video_id>.glb`** with the following properties:

* **Single mesh**: export one triangle mesh without node transforms.

* **Real size**: apply the estimated or optimized scale to the vertices and use meters.

* **Centroid at the origin**: use the area-weighted surface centroid returned by
  `trimesh.load(path, force="mesh").centroid`. Each coordinate must be **within 1 mm of zero**.

#### 5.2.1 Centering the mesh and updating its trajectory

The JSON poses must refer to the vertices stored in the GLB.

If your estimated poses refer to an uncentered mesh:

1. Subtract its centroid `c` from the vertices.
2. Add `R @ c` to each translation.

This preserves the object's position in the camera frame:

```text
R @ (v - c) + (t + R @ c) = R @ v + t
```

Apply this correction **before saving `object.transl`** in the JSON ([§5.1](#51-json-file-hand-and-object-trajectory)):

```python
import numpy as np, trimesh
mesh = trimesh.load("my_object.glb", force="mesh")   # the mesh your poses R (F,3,3), t (F,3) refer to
c = mesh.centroid                                     # area-weighted surface centroid, object frame
mesh.vertices -= c                                    # centroid -> origin
t = t + np.einsum("fij,j->fi", R, c)                  # R (v - c) + t_new = R v + t, for every frame
mesh.export(f"results/{video_id}.glb")                # export with trimesh (no axis conversion)

# check: the saved mesh is centered (tolerance 1 mm)
assert np.abs(trimesh.load(f"results/{video_id}.glb", force="mesh").centroid).max() < 1e-3
```

> Export the GLB with **`trimesh`**, as above. Other exporters may convert axes or add node
> transforms; either change can make the saved mesh inconsistent with the JSON poses.

### 5.3 Example files and self-checks

`examples/` contains a short clip from a public dataset (not one of the course videos; source
and license in `examples/ATTRIBUTION.md`) and a complete, valid output for it:

```
examples/
├── data/example_01/frames/       # input: 000000.jpg … 000149.jpg (640 × 480), same layout as §3
├── results/example_01.json       # output in the format of §5.1
├── results/example_01.glb        # object mesh, centered at its centroid (§5.2)
└── overlays/example_01.mp4/.jpg  # the output drawn on the input (tools/overlay.py)
```

The output was made from the dataset's annotations **with errors added on purpose** (hand pose
and depth, object size, rotation and depth, focal length), so it lies close to, but visibly off,
the hand and the object. It illustrates the file format, not a target accuracy. The hand vertices
and joints were computed from the included MANO parameters, and the mesh is centered at its
centroid.

#### Check the MANO output and mesh centroid

Use the example files to check your MANO setup and mesh loading:

```python
import json, numpy as np, torch, smplx, trimesh
ex = json.load(open("examples/results/example_01.json"))

# 1) MANO settings: your forward pass must reproduce the example vertices
mano = smplx.MANO("mano_models", is_rhand=True, use_pca=False, flat_hand_mean=True)
pose = torch.tensor(ex["mano"]["pose"], dtype=torch.float32)
out = mano(global_orient=pose[:, :3], hand_pose=pose[:, 3:],
           betas=torch.tensor(ex["mano"]["betas"], dtype=torch.float32)[None].expand(len(pose), -1),
           transl=torch.tensor(ex["mano"]["transl"], dtype=torch.float32))
print(np.abs(out.vertices.detach().numpy() - np.array(ex["mano"]["vertices"])).max())   # < 1e-4

# 2) mesh centered at its centroid
print(trimesh.load("examples/results/example_01.glb", force="mesh").centroid)           # ~ (0, 0, 0)
```

#### Check the rotation convention

Decode `ex["object"]["rot6d"][0]` using the code in §5.1.2. The result should match the matrix
below to four decimal places:

```
R_0 = [[ 0.6139, -0.1837, -0.7677],
       [-0.3414,  0.8151, -0.4680],
       [ 0.7118,  0.5494,  0.4376]]
```

#### Check the example overlay

`examples/overlays/example_01.mp4` shows the example output drawn on its input frames (left:
input, right: overlay; object in green, hand vertices in cyan with a 21-joint skeleton).
`tools/overlay.py` reproduces it (`pip install numpy trimesh pillow imageio imageio-ffmpeg`):

```bash
python tools/overlay.py --frames examples/data/example_01/frames --json examples/results/example_01.json \
                        --out overlays/example_01.mp4 --side-by-side --grid
```

![example overlay](examples/overlays/example_01.jpg)

#### Check your results on the input frames

Reload your saved JSON and GLB files, project their vertices onto the input frames, and inspect
the overlays. The code below draws the object in green and the hand in cyan for the first,
middle, and last frames. Check that they align with the visible object and hand.

```python
import json, numpy as np, trimesh
from PIL import Image, ImageDraw
d = json.load(open(f"results/{video_id}.json"))
K = np.array(d["intrinsics"]); V = np.array(d["mano"]["vertices"])
mesh = trimesh.load(f"results/{video_id}.glb", force="mesh")
rot6d = np.array(d["object"]["rot6d"]); t_obj = np.array(d["object"]["transl"])
b1 = rot6d[:, :3] / np.linalg.norm(rot6d[:, :3], axis=1, keepdims=True)
b2 = rot6d[:, 3:] - (b1 * rot6d[:, 3:]).sum(1, keepdims=True) * b1
b2 /= np.linalg.norm(b2, axis=1, keepdims=True); R = np.stack([b1, b2, np.cross(b1, b2)], 1)
proj = lambda X: X[:, :2] / X[:, 2:] * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
for t in [0, d["num_frames"] // 2, d["num_frames"] - 1]:
    img = Image.open(f"data/{video_id}/frames/{t:06d}.jpg").convert("RGB"); dr = ImageDraw.Draw(img)
    for u, v in proj(mesh.vertices @ R[t].T + t_obj[t]): dr.point((u, v), fill=(0, 255, 0))
    for u, v in proj(V[t]):                               dr.point((u, v), fill=(0, 200, 255))
    img.save(f"check_{video_id}_{t:06d}.jpg")
```

If the overlays do not align, check the camera intrinsics, coordinate conventions, units,
rotation layout, mesh centering, and MANO settings, as well as the reconstruction itself.

---

## 6. Leaderboard

**Stay tuned! We will launch the leaderboard within 2 weeks.**

<!--
* **Where**: Gradescope, assignment "Final Project Leaderboard" [TBD: course link / entry code].
* **What to upload**: all 12 result files, either
  * as **one zip** (`results.zip`, the files at the top level of the zip, no sub-folders; upload the
    zip itself), or
  * as **individual files**, selecting all `.json` and `.glb` files together in one upload.

  File names must be **exactly** `<video_id>.json` and `<video_id>.glb`, e.g. `easy_01.json` +
  `easy_01.glb`, …, `hard_02.json` + `hard_02.glb` (lower case, as the video folders in `data/`).
  Every `.json` needs its `.glb` with the same name in the same upload. A missing or misnamed file
  means that video is missing.
* **How often**: no limit on the number of uploads. Each upload is graded automatically within a
  few minutes.
* **What you see after an upload**:
  * a format check for each video: `OK`, or the reason it was rejected (missing file, wrong shape,
    NaN, values that are not plausible in meters, mesh not centred, …);
  * your scores for each video and your overall score.
* **What the leaderboard shows**: the scores of everyone's latest upload under a pseudonym you
  choose at upload time, sortable by each column [TBD: columns].
* **Missing or rejected videos** get the lowest possible score for that video [TBD].
* **Scores on the leaderboard are feedback**: they are computed on a part of the frames and compared
  with the class at the time of the last update [TBD]. The final grade is computed after the
  deadline on all frames from your LMS submission (§7).
* Do not try to probe or reverse-engineer the ground truth through the leaderboard.
-->

## 7. Submission

**Deadline: 12/7 23:59.**

| Where | What |
|---|---|
| **Leaderboard** | `results.zip` — [TBD: link] |
| **LMS** | `report.pdf`, `main.zip` (zip only the `main/` folder) |
