# Submission format (format_version 1.0, DRAFT)

Per video: `results/<video_id>.json` + `results/<video_id>.glb`. Every convention below is fixed;
a file that ignores one is usually still readable, but it describes a **different** hand or object
and is graded as such. Check your results by projecting them onto the frames (§7).

## 1. Common conventions

| Item | Convention |
|---|---|
| Camera frame | **OpenCV**: +x right, +y down, +z forward (into the scene). The camera is static; there is no world frame |
| Length unit | **metres** (vertices, joints, translations) |
| Intrinsics | **pixels**, for the resolution of the given frames (`width`×`height` in `meta.json`) |
| Frames | `F = num_frames` of the video; entry `t` corresponds to `frames/{t:06d}.jpg`; **every** frame is filled (JSON has no NaN: interpolate frames where your method failed) |
| Hand | right hand, `MANO_RIGHT.pkl`, `use_pca=False`, `flat_hand_mean=True` |
| Numbers | plain JSON numbers; round to 5 decimals (0.01 mm) to keep files small |

## 2. JSON keys

```text
{
  "format_version": "1.0",
  "video_id":   "easy_01",
  "num_frames": F,
  "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],        # [3,3] your estimate, pixels
  "mano": {
    "pose":     [F][48],      # axis-angle: global_orient (3) + 15 finger joints (45, parent-relative)
    "transl":   [F][3],       # metres, the `transl` argument of smplx.MANO
    "betas":    [10],         # one hand shape for the whole video
    "vertices": [F][778][3],  # smplx.MANO(...) output.vertices, camera frame, metres
    "joints":   [F][16][3]    # smplx.MANO(...) output.joints,   camera frame, metres
  },
  "object": {
    "mesh":   "easy_01.glb",  # file name of the mesh next to this JSON
    "rot6d":  [F][6],         # rotation, 6D representation, ROW layout (§4)
    "transl": [F][3]          # metres, position of the mesh origin = mesh centroid (§5)
  }
}
```

`examples/example_01.json` (+ `example_01.glb`) is a valid file with this structure, F = 2. Its numbers are
placeholders: the vertices and joints are **not** the MANO output of its parameters.

## 3. Hand (MANO)

`vertices` and `joints` must be **exactly** the output of this forward pass with your
`pose`, `betas`, `transl` (the TAs re-run it to check):

```python
import smplx, torch
mano = smplx.MANO("mano_models", is_rhand=True, use_pca=False, flat_hand_mean=True)
pose   = torch.as_tensor(pose,   dtype=torch.float32)          # (F, 48)
betas  = torch.as_tensor(betas,  dtype=torch.float32)          # (10,)
transl = torch.as_tensor(transl, dtype=torch.float32)          # (F, 3)
out = mano(global_orient=pose[:, :3], hand_pose=pose[:, 3:],
           betas=betas[None].expand(len(pose), -1), transl=transl)
vertices = out.vertices   # (F, 778, 3)
joints   = out.joints     # (F, 16, 3)
```

* `flat_hand_mean=True`: `hand_pose = 0` is the flat open hand. Many models (and smplx's default)
  use `False`, where the mean pose is added to your numbers. Convert if your model uses another
  setting, then check the projection.
* `transl` is added to all vertices after the hand is rotated about its wrist joint (smplx). It is
  **not** the wrist position (the wrist is ~9.6 cm from the MANO origin).
* `pose` is axis-angle (radians). If your model gives rotation matrices (e.g. HaPTIC:
  `(F, 16, 3, 3)`), convert them, e.g. with `scipy.spatial.transform.Rotation.from_matrix(...).as_rotvec()`.
* `betas`: one vector for the video (models that predict per-frame shape: pick or average one,
  and use it in the forward pass).

## 4. Object rotation: 6D representation, ROW layout

The 6 numbers are the **first two rows** of the rotation matrix R (as in PyTorch3D
`matrix_to_rotation_6d` / `rotation_6d_to_matrix`):

```python
rot6d = R[:, :2, :].reshape(F, 6)            # R: (F, 3, 3) object -> camera

# inverse (what the grader does)
a1, a2 = rot6d[:, :3], rot6d[:, 3:]
b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
b2 = a2 - (b1 * a2).sum(1, keepdims=True) * b1
b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
R = np.stack([b1, b2, np.cross(b1, b2)], axis=1)  # rows
```

Using the first two **columns** instead gives the transposed (inverse) rotation and a wrong object.

## 5. Object mesh and translation: centred at the centroid

* `<video_id>.glb`: the **final** mesh, real size (scale applied), a **single** triangle mesh with no
  node transform, **translated so that its centroid is at the origin**:
  `trimesh.load(path, force="mesh").centroid ≈ (0, 0, 0)` (within 1 mm).
* Pose: `X_cam = R @ v + t` for every GLB vertex `v`. Because the mesh is centred, `t` is the
  centroid's position in the camera frame.

If your tracker gives a pose for an uncentred mesh (e.g. FoundationPose `ob_in_cam`), centre the
mesh and move the translation accordingly (the object stays where it is in the camera):

```python
import numpy as np, trimesh
mesh = trimesh.load("my_object.glb", force="mesh")         # the mesh your poses refer to
c = mesh.centroid
mesh.vertices -= c                                          # centroid -> origin
t = t + np.einsum("fij,j->fi", R, c)                        # R (F,3,3), t (F,3): R(v - c) + t' = R v + t
mesh.export(f"results/{video_id}.glb")                      # export with trimesh (no axis conversion)
```

Do not export the GLB from Blender or other tools that convert axes (Z-up / Y-up) or add node
transforms: the vertices must be in the same object frame as `R`, `t`.

## 6. Writing the JSON

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
    json.dump(sub, f, separators=(",", ":"), allow_nan=False)   # allow_nan=False: fail on NaN
```

## 7. Check before submitting

Project your result onto the frames **from the files you wrote** (read them back, do not reuse the
arrays in memory) and look at them:

```python
import json, numpy as np, trimesh
from PIL import Image, ImageDraw
d = json.load(open(f"results/{video_id}.json"))
K = np.array(d["intrinsics"]); V = np.array(d["mano"]["vertices"])
mesh = trimesh.load(f"results/{video_id}.glb", force="mesh")
a1, a2 = np.array(d["object"]["rot6d"])[:, :3], np.array(d["object"]["rot6d"])[:, 3:]
b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True); b2 = a2 - (b1 * a2).sum(1, keepdims=True) * b1
b2 /= np.linalg.norm(b2, axis=1, keepdims=True); R = np.stack([b1, b2, np.cross(b1, b2)], 1)
t_obj = np.array(d["object"]["transl"])
proj = lambda X: X[:, :2] / X[:, 2:] * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
for t in [0, d["num_frames"] // 2, d["num_frames"] - 1]:
    img = Image.open(f"data/{video_id}/frames/{t:06d}.jpg").convert("RGB"); dr = ImageDraw.Draw(img)
    for u, v in proj(mesh.vertices @ R[t].T + t_obj[t]): dr.point((u, v), fill=(0, 255, 0))
    for u, v in proj(V[t]):                               dr.point((u, v), fill=(0, 200, 255))
    img.save(f"check_{video_id}_{t:06d}.jpg")
```

Green (object) and cyan (hand) must lie on the object and the hand in the video. If they do not, a
convention above is wrong (camera frame, units, rotation layout, mesh centring, MANO settings).
