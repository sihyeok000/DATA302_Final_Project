"""Draw your results on the input frames and save an overlay video (mp4). Example: README §5.3.

    python tools/overlay.py --video_id easy_01   # data/easy_01/frames + results/easy_01.json -> overlays/easy_01.mp4
    python tools/overlay.py --all                # every results/*.json

Files are read exactly as the grader reads them, so if the drawing does not lie on the hand and the
object, one of the conventions in README §5 is wrong.
"""
import argparse
import glob
import json
import os

import imageio
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter

DATA_DIR, RESULTS_DIR, OUT_DIR = "data", "results", "overlays"     # data/<video_id>/frames/{t:06d}.jpg

# 21 joints in OpenPose order = MANO output.joints (16) + 5 fingertip vertices, reordered
TIP_VERTS = [745, 317, 444, 556, 673]
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
         (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
FINGER_COLORS = [(255, 70, 70), (255, 165, 0), (240, 230, 0), (60, 210, 90), (40, 150, 255)]
OBJ_RGB, HAND_RGB = np.array([40, 230, 90]), (90, 230, 255)


def rot6d_rows_to_matrix(x):
    """(F,6) first two ROWS of R -> (F,3,3) (README §5.1.2)."""
    a1, a2 = x[:, :3], x[:, 3:]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(1, keepdims=True) * b1
    b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def load_result(json_path, glb_path=None):
    with open(json_path) as f:
        d = json.load(f)
    glb_path = glb_path or os.path.join(os.path.dirname(json_path), d["object"]["mesh"])
    mesh = trimesh.load(glb_path, force="mesh")
    if np.abs(mesh.centroid).max() > 1e-3:
        print(f"WARNING {os.path.basename(glb_path)}: centroid is {np.round(1000 * mesh.centroid, 1)} mm "
              f"from the origin; the grader rejects this (README §5.2)")
    verts = np.asarray(d["mano"]["vertices"], np.float64)
    joints16 = np.asarray(d["mano"]["joints"], np.float64)
    return {
        "video_id": d["video_id"],
        "F": d["num_frames"],
        "K": np.asarray(d["intrinsics"], np.float64),
        "hand_verts": verts,
        "hand_joints": np.concatenate([joints16, verts[:, TIP_VERTS]], axis=1)[:, MANO_TO_OPENPOSE],
        "R": rot6d_rows_to_matrix(np.asarray(d["object"]["rot6d"], np.float64)),
        "t": np.asarray(d["object"]["transl"], np.float64),
        "obj_verts": np.asarray(mesh.vertices, np.float64),
        "obj_faces": np.asarray(mesh.faces),
    }


def project(X, K):
    """Camera-frame points (...,3) -> pixels (...,2) and a mask of points in front of the camera."""
    z = X[..., 2]
    in_front = z > 1e-6
    uv = X[..., :2] / np.where(in_front, z, np.nan)[..., None]
    return uv * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]], in_front


def draw_object(img, X, faces, K, alpha=0.55):
    """Semi-transparent mesh, faces painted far to near and shaded by how much they face the camera."""
    uv, ok = project(X, K)
    tri = X[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    centre = tri.mean(1)
    view = -centre / (np.linalg.norm(centre, axis=1, keepdims=True) + 1e-12)
    shade = 0.35 + 0.65 * np.abs((normal * view).sum(1))

    layer, mask = Image.new("RGB", img.size), Image.new("L", img.size, 0)
    draw_layer, draw_mask = ImageDraw.Draw(layer), ImageDraw.Draw(mask)
    order = np.argsort(-centre[:, 2])
    for f in order[ok[faces].all(1)[order]]:
        poly = [tuple(p) for p in uv[faces[f]]]
        draw_layer.polygon(poly, fill=tuple(int(c) for c in OBJ_RGB * shade[f]))
        draw_mask.polygon(poly, fill=255)
    img = Image.composite(Image.blend(img, layer, alpha), img, mask)
    outline = Image.new("RGB", img.size, tuple(int(c) for c in OBJ_RGB))
    img.paste(outline, mask=mask.filter(ImageFilter.FIND_EDGES))
    return img


def draw_hand(img, verts, joints, K):
    """MANO vertices as dots and the 21-joint skeleton, one colour per finger."""
    draw = ImageDraw.Draw(img)
    uv, ok = project(verts, K)
    for p in uv[ok]:
        draw.point(tuple(p), fill=HAND_RGB)
    uv, ok = project(joints, K)
    for i, (a, b) in enumerate(BONES):
        if ok[a] and ok[b]:
            draw.line([tuple(uv[a]), tuple(uv[b])], fill=FINGER_COLORS[i // 4], width=2)
    return img


def draw_frame(img, r, t):
    X = r["obj_verts"] @ r["R"][t].T + r["t"][t]
    img = draw_object(img, X, r["obj_faces"], r["K"])
    img = draw_hand(img, r["hand_verts"][t], r["hand_joints"][t], r["K"])
    ImageDraw.Draw(img).text((6, 4), f"{r['video_id']}  t={t}", fill=(255, 255, 255))
    return img


def save_grid(tiles, path, width=480):
    tiles = [im.resize((width, round(im.height * width / im.width))) for im in tiles]
    h = tiles[0].height
    sheet = Image.new("RGB", (4 * width, (len(tiles) + 3) // 4 * h))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i % 4 * width, i // 4 * h))
    sheet.save(path, quality=90)


def overlay_video(frames_dir, json_path, glb_path, out, fps=30, side_by_side=False, grid=False):
    r = load_result(json_path, glb_path)
    n_files = len(glob.glob(os.path.join(frames_dir, "*.jpg")))
    missing = [t for t in range(r["F"]) if not os.path.isfile(os.path.join(frames_dir, f"{t:06d}.jpg"))]
    if n_files != r["F"] or missing:
        raise SystemExit(f"{frames_dir} has {n_files} frames but {json_path} has num_frames = {r['F']}"
                         + (f" (missing {missing[0]:06d}.jpg)" if missing else "")
                         + ": every frame needs a result (README §5, Conventions overview)")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    writer = imageio.get_writer(out, "ffmpeg", fps=fps, codec="libx264", quality=None, pixelformat="yuv420p",
                                macro_block_size=2, output_params=["-crf", "20"])
    grid_frames = set(np.linspace(0, r["F"] - 1, min(8, r["F"])).round().astype(int))
    tiles = []
    try:
        for t in range(r["F"]):
            src = Image.open(os.path.join(frames_dir, f"{t:06d}.jpg")).convert("RGB")
            img = draw_frame(src, r, t)
            if grid and t in grid_frames:
                tiles.append(img)
            if side_by_side:
                both = Image.new("RGB", (2 * img.width, img.height))
                both.paste(src, (0, 0))
                both.paste(img, (img.width, 0))
                img = both
            writer.append_data(np.asarray(img))
    finally:
        writer.close()
    if tiles:
        save_grid(tiles, os.path.splitext(out)[0] + ".jpg")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video_id", help="use the default paths for this video id")
    ap.add_argument("--all", action="store_true", help=f"every {RESULTS_DIR}/*.json")
    ap.add_argument("--frames", help=f"input frames folder (default {DATA_DIR}/<video_id>/frames)")
    ap.add_argument("--json", help=f"result json (default {RESULTS_DIR}/<video_id>.json)")
    ap.add_argument("--glb", help="object mesh (default: the 'object.mesh' file next to the json)")
    ap.add_argument("--out", help=f"output video (default {OUT_DIR}/<video_id>.mp4)")
    ap.add_argument("--fps", type=float, default=30, help="frame rate of the output video (default 30)")
    ap.add_argument("--side-by-side", action="store_true", help="input and overlay next to each other")
    ap.add_argument("--grid", action="store_true", help="also save an 8-frame overview .jpg next to the video")
    args = ap.parse_args()

    if args.all:
        jobs = [(os.path.basename(p)[:-5], p) for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))]
        jobs = [(os.path.join(DATA_DIR, vid, "frames"), p, None, os.path.join(OUT_DIR, f"{vid}.mp4"))
                for vid, p in jobs]
    elif args.video_id or args.json:
        vid = args.video_id or os.path.basename(args.json)[:-5]
        jobs = [(args.frames or os.path.join(DATA_DIR, vid, "frames"),
                 args.json or os.path.join(RESULTS_DIR, f"{vid}.json"),
                 args.glb,
                 args.out or os.path.join(OUT_DIR, f"{vid}.mp4"))]
    else:
        ap.error("give --video_id, --json or --all")
    for frames_dir, json_path, glb_path, out in jobs:
        print("wrote", overlay_video(frames_dir, json_path, glb_path, out, args.fps, args.side_by_side, args.grid))


if __name__ == "__main__":
    main()
