"""Draw your results on the input video and save an overlay video (mp4).

The files are read exactly as the grader reads them (README §6): rot6d row layout, .glb centred at its
centroid, the intrinsics in your JSON. If the drawing does not lie on the hand and the object in the
video, a convention is wrong (units, camera frame, rotation layout, mesh centring, MANO settings, K).

    pip install numpy trimesh pillow imageio imageio-ffmpeg

    # one video, default paths (run from the repository root)
    #   video  data/videos/easy_01.mp4    json  results/easy_01.json
    #   glb    results/easy_01.glb        out   overlays/easy_01.mp4
    python tools/overlay.py --video_id easy_01

    # every results/*.json
    python tools/overlay.py --all

    # explicit paths
    python tools/overlay.py --video data/videos/easy_01.mp4 --json results/easy_01.json \
                            --glb results/easy_01.glb --out overlays/easy_01.mp4

    # options: --side-by-side (input | overlay), --grid (also save an 8-frame overview .jpg)

Object: shaded, semi-transparent green mesh with its outline. Hand: MANO vertices (cyan dots) and
the 21-joint skeleton (one colour per finger).
"""
import argparse
import glob
import json
import os

import imageio
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter

DEFAULT_VIDEO_DIR, DEFAULT_RESULTS_DIR, DEFAULT_OUT_DIR = "data/videos", "results", "overlays"

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


def load_result(json_path, glb_path=None):
    with open(json_path) as f:
        d = json.load(f)
    glb_path = glb_path or os.path.join(os.path.dirname(json_path), d["object"]["mesh"])
    mesh = trimesh.load(glb_path, force="mesh")
    if np.abs(mesh.centroid).max() > 1e-3:
        print(f"WARNING {os.path.basename(glb_path)}: centroid is {np.round(1000 * mesh.centroid, 1)} mm from "
              f"the origin; the grader rejects this (README §6.6)")
    V = np.asarray(d["mano"]["vertices"], np.float64)
    J16 = np.asarray(d["mano"]["joints"], np.float64)
    return {
        "video_id": d["video_id"], "F": d["num_frames"], "K": np.asarray(d["intrinsics"], np.float64),
        "hand_verts": V,
        "hand_joints": np.concatenate([J16, V[:, TIP_VERTS]], axis=1)[:, MANO_TO_OPENPOSE],
        "R": rot6d_rows_to_matrix(np.asarray(d["object"]["rot6d"], np.float64)),
        "t": np.asarray(d["object"]["transl"], np.float64),
        "obj_verts": np.asarray(mesh.vertices, np.float64), "obj_faces": np.asarray(mesh.faces),
    }


def project(X, K):
    z = X[..., 2]
    uv = X[..., :2] / np.where(z > 1e-6, z, np.nan)[..., None]
    return uv * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]], z > 1e-6


def draw_frame(img, r, t, alpha=0.55):
    """img: PIL RGB frame t. Returns the frame with the reconstruction drawn on top."""
    K = r["K"]
    # object: faces painted far -> near, shaded by how much they face the camera
    Xc = r["obj_verts"] @ r["R"][t].T + r["t"][t]
    uv, ok = project(Xc, K)
    F = r["obj_faces"]
    tri = Xc[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    centre = tri.mean(1)
    view = -centre / (np.linalg.norm(centre, axis=1, keepdims=True) + 1e-12)
    shade = 0.35 + 0.65 * np.abs((n * view).sum(1))
    keep = ok[F].all(1)
    order = np.argsort(-centre[:, 2])
    layer, mask = Image.new("RGB", img.size), Image.new("L", img.size, 0)
    dl, dm = ImageDraw.Draw(layer), ImageDraw.Draw(mask)
    for f in order[keep[order]]:
        poly = [tuple(p) for p in uv[F[f]]]
        dl.polygon(poly, fill=tuple(int(c) for c in OBJ_RGB * shade[f]))
        dm.polygon(poly, fill=255)
    img = Image.composite(Image.blend(img, layer, alpha), img, mask)
    img.paste(Image.new("RGB", img.size, tuple(int(c) for c in OBJ_RGB)), mask=mask.filter(ImageFilter.FIND_EDGES))
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


def overlay_video(video, json_path, glb, out, side_by_side=False, grid=False):
    r = load_result(json_path, glb)
    reader = imageio.get_reader(video, "ffmpeg")
    fps = reader.get_meta_data().get("fps", 30)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    writer = imageio.get_writer(out, "ffmpeg", fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
                                macro_block_size=2)
    picks, tiles = set(np.linspace(0, r["F"] - 1, min(8, r["F"])).round().astype(int)), []
    n = 0
    try:
        for t, frame in enumerate(reader):
            if t >= r["F"]:
                n = t + 1
                continue                               # count the extra frames, report below
            src = Image.fromarray(frame).convert("RGB")
            img = draw_frame(src, r, t)
            ImageDraw.Draw(img).text((6, 4), f"{r['video_id']}  t={t}", fill=(255, 255, 255))
            if grid and t in picks:
                tiles.append(img.resize((480, round(img.height * 480 / img.width))))
            if side_by_side:
                both = Image.new("RGB", (2 * img.width, img.height))
                both.paste(src, (0, 0))
                both.paste(img, (img.width, 0))
                img = both
            writer.append_data(np.asarray(img))
            n = t + 1
    finally:
        writer.close()
        reader.close()
    if n != r["F"]:
        raise SystemExit(f"{video} has {n} frames but {json_path} has num_frames = {r['F']}: "
                         f"every frame of the video needs a result (README §6.2)")
    if grid and tiles:
        sheet = Image.new("RGB", (4 * 480, ((len(tiles) + 3) // 4) * tiles[0].height))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % 4) * 480, (i // 4) * tiles[0].height))
        sheet.save(os.path.splitext(out)[0] + ".jpg", quality=90)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video_id", help="use the default paths for this video id")
    ap.add_argument("--all", action="store_true", help=f"every {DEFAULT_RESULTS_DIR}/*.json")
    ap.add_argument("--video", help=f"input video (default {DEFAULT_VIDEO_DIR}/<video_id>.mp4)")
    ap.add_argument("--json", help=f"result json (default {DEFAULT_RESULTS_DIR}/<video_id>.json)")
    ap.add_argument("--glb", help="object mesh (default: the 'object.mesh' file next to the json)")
    ap.add_argument("--out", help=f"output video (default {DEFAULT_OUT_DIR}/<video_id>.mp4)")
    ap.add_argument("--side-by-side", action="store_true", help="input and overlay next to each other")
    ap.add_argument("--grid", action="store_true", help="also save an 8-frame overview image next to the video")
    args = ap.parse_args()

    if args.all:
        vids = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(DEFAULT_RESULTS_DIR, "*.json")))
    elif args.video_id or args.json:
        vids = [args.video_id or os.path.basename(args.json)[:-5]]
    else:
        ap.error("give --video_id, --json or --all")
    for vid in vids:
        one = len(vids) == 1
        video = (args.video if one else None) or os.path.join(DEFAULT_VIDEO_DIR, f"{vid}.mp4")
        json_path = (args.json if one else None) or os.path.join(DEFAULT_RESULTS_DIR, f"{vid}.json")
        out = (args.out if one else None) or os.path.join(DEFAULT_OUT_DIR, f"{vid}.mp4")
        print("wrote", overlay_video(video, json_path, args.glb if one else None, out, args.side_by_side, args.grid))


if __name__ == "__main__":
    main()
