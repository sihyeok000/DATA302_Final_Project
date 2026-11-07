"""Draw your results on the video frames, reading the files exactly as the grader does.

    python tools/overlay.py                          # every results/<video_id>.json, 8 frames each
    python tools/overlay.py --video easy_01 --num 12
    python tools/overlay.py --every 1 --frames-out   # also one image per frame (for a video)

Writes overlays/<video_id>.jpg (a grid of frames). Object: shaded, semi-transparent mesh (green).
Hand: MANO vertices (cyan dots) and the 21-joint skeleton (one colour per finger).
If the drawing does not lie on the hand and the object in the video, a convention is wrong
(README §6): units, camera frame, rotation layout, mesh centring, MANO settings, intrinsics.

Needs numpy, trimesh, Pillow.
"""
import argparse
import glob
import json
import os

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter

# 21 joints in OpenPose order = MANO forward output.joints (16) + fingertip vertices, reordered
TIP_VERTS = [745, 317, 444, 556, 673]
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
         (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
FINGER_COLORS = [(255, 70, 70), (255, 165, 0), (240, 230, 0), (60, 210, 90), (40, 150, 255)]
OBJ_RGB, HAND_RGB = np.array([40, 230, 90]), (90, 230, 255)


def rot6d_rows_to_matrix(x):
    """(F,6) first two ROWS of R -> (F,3,3), the README §6.5 convention."""
    a1, a2 = x[:, :3], x[:, 3:]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(1, keepdims=True) * b1
    b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def load_result(json_path):
    with open(json_path) as f:
        d = json.load(f)
    mesh = trimesh.load(os.path.join(os.path.dirname(json_path), d["object"]["mesh"]), force="mesh")
    V = np.asarray(d["mano"]["vertices"], np.float64)
    J16 = np.asarray(d["mano"]["joints"], np.float64)
    return {
        "video_id": d["video_id"], "F": d["num_frames"], "K": np.asarray(d["intrinsics"], np.float64),
        "hand_verts": V,
        "hand_joints": np.concatenate([J16, V[:, TIP_VERTS]], axis=1)[:, MANO_TO_OPENPOSE],
        "R": rot6d_rows_to_matrix(np.asarray(d["object"]["rot6d"], np.float64)),
        "t": np.asarray(d["object"]["transl"], np.float64),
        "obj_verts": np.asarray(mesh.vertices, np.float64), "obj_faces": np.asarray(mesh.faces),
        "centroid": np.asarray(mesh.centroid),
    }


def project(X, K):
    z = X[..., 2]
    uv = X[..., :2] / np.where(z > 1e-6, z, np.nan)[..., None]
    return uv * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]], z > 1e-6


def draw_frame(img, r, t, alpha=0.55):
    K = r["K"]
    # object: faces painted far -> near, shaded by how much they face the camera
    Xc = r["obj_verts"] @ r["R"][t].T + r["t"][t]
    uv, ok = project(Xc, K)
    F = r["obj_faces"]
    tri = Xc[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    view = -tri.mean(1) / (np.linalg.norm(tri.mean(1), axis=1, keepdims=True) + 1e-12)
    shade = 0.35 + 0.65 * np.abs((n * view).sum(1))
    keep = ok[F].all(1)
    order = np.argsort(-tri[:, :, 2].mean(1))
    layer = Image.new("RGB", img.size)
    mask = Image.new("L", img.size, 0)
    dl, dm = ImageDraw.Draw(layer), ImageDraw.Draw(mask)
    for f in order[keep[order]]:
        poly = [tuple(p) for p in uv[F[f]]]
        dl.polygon(poly, fill=tuple(int(c) for c in OBJ_RGB * shade[f]))
        dm.polygon(poly, fill=255)
    img = Image.composite(Image.blend(img, layer, alpha), img, mask)
    edge = mask.filter(ImageFilter.FIND_EDGES)
    img.paste(Image.new("RGB", img.size, tuple(OBJ_RGB)), mask=edge)
    # hand
    d = ImageDraw.Draw(img)
    for (u, v), good in zip(*project(r["hand_verts"][t], K)):
        if good:
            d.point((u, v), fill=HAND_RGB)
    uv, ok = project(r["hand_joints"][t], K)
    for i, (a, b) in enumerate(BONES):
        if ok[a] and ok[b]:
            d.line([tuple(uv[a]), tuple(uv[b])], fill=FINGER_COLORS[i // 4], width=2)
    return img


def overlay_video(json_path, data_root, out_dir, num=8, every=None, frames_out=False, tile_width=480):
    r = load_result(json_path)
    vid, F = r["video_id"], r["F"]
    frames = sorted(glob.glob(os.path.join(data_root, vid, "frames", "*.jpg")) +
                    glob.glob(os.path.join(data_root, vid, "frames", "*.png")))
    if len(frames) != F:
        raise SystemExit(f"{vid}: {len(frames)} frames in {data_root}/{vid}/frames but num_frames = {F}")
    if np.abs(r["centroid"]).max() > 1e-3:
        print(f"WARNING {vid}: the .glb centroid is {np.round(1000 * r['centroid'], 1)} mm from the origin (README §6.6)")
    ts = list(range(0, F, every)) if every else sorted(set(np.linspace(0, F - 1, min(num, F)).round().astype(int)))
    tiles = []
    for t in ts:
        img = draw_frame(Image.open(frames[t]).convert("RGB"), r, t)
        if frames_out:
            os.makedirs(os.path.join(out_dir, vid), exist_ok=True)
            img.save(os.path.join(out_dir, vid, f"{t:06d}.jpg"), quality=90)
        tile = img.resize((tile_width, round(img.height * tile_width / img.width)))
        ImageDraw.Draw(tile).text((6, 4), f"{vid}  t={t}", fill=(255, 255, 255))
        tiles.append(tile)
    shown = tiles if len(tiles) <= 16 else [tiles[i] for i in np.linspace(0, len(tiles) - 1, 16).round().astype(int)]
    cols = 4
    rows = (len(shown) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_width, rows * shown[0].height))
    for i, tile in enumerate(shown):
        sheet.paste(tile, ((i % cols) * tile_width, (i // cols) * shown[0].height))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{vid}.jpg")
    sheet.save(path, quality=90)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", help="folder with <video_id>.json + .glb")
    ap.add_argument("--data", default="data", help="folder with <video_id>/frames/")
    ap.add_argument("--out", default="overlays")
    ap.add_argument("--video", nargs="*", help="video ids (default: every .json in --results)")
    ap.add_argument("--num", type=int, default=8, help="number of evenly spaced frames in the grid")
    ap.add_argument("--every", type=int, default=None, help="draw every k-th frame instead of --num")
    ap.add_argument("--frames-out", action="store_true", help="also save one image per drawn frame")
    args = ap.parse_args()
    vids = args.video or sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(args.results, "*.json")))
    for vid in vids:
        print("wrote", overlay_video(os.path.join(args.results, f"{vid}.json"), args.data, args.out,
                                     args.num, args.every, args.frames_out))


if __name__ == "__main__":
    main()
