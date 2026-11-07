"""Draw your results on the input video and save an overlay video (usage: README §6.8).

Files are read exactly as the grader reads them, so if the drawing does not lie on the hand and the
object, one of the conventions in README §6 is wrong.
"""
import argparse
import glob
import json
import os

import imageio
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter

VIDEO_DIR, RESULTS_DIR, OUT_DIR = "data/videos", "results", "overlays"

# 21 joints in OpenPose order = MANO output.joints (16) + 5 fingertip vertices, reordered
TIP_VERTS = [745, 317, 444, 556, 673]
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
         (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
FINGER_COLORS = [(255, 70, 70), (255, 165, 0), (240, 230, 0), (60, 210, 90), (40, 150, 255)]
OBJ_RGB, HAND_RGB = np.array([40, 230, 90]), (90, 230, 255)


def rot6d_rows_to_matrix(x):
    """(F,6) first two ROWS of R -> (F,3,3) (README §6.5)."""
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
              f"from the origin; the grader rejects this (README §6.6)")
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


def overlay_video(video, json_path, glb_path, out, side_by_side=False, grid=False):
    r = load_result(json_path, glb_path)
    reader = imageio.get_reader(video, "ffmpeg")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    writer = imageio.get_writer(out, "ffmpeg", fps=reader.get_meta_data().get("fps", 30), codec="libx264",
                                quality=None, pixelformat="yuv420p", macro_block_size=2,
                                output_params=["-crf", "20"])
    grid_frames = set(np.linspace(0, r["F"] - 1, min(8, r["F"])).round().astype(int))
    tiles, n_frames = [], 0
    try:
        for t, frame in enumerate(reader):
            n_frames += 1
            if t >= r["F"]:
                continue                       # keep counting, reported below
            src = Image.fromarray(frame).convert("RGB")
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
        reader.close()
    if n_frames != r["F"]:
        raise SystemExit(f"{video} has {n_frames} frames but {json_path} has num_frames = {r['F']}: "
                         f"every frame of the video needs a result (README §6.2)")
    if tiles:
        save_grid(tiles, os.path.splitext(out)[0] + ".jpg")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video_id", help="use the default paths for this video id")
    ap.add_argument("--all", action="store_true", help=f"every {RESULTS_DIR}/*.json")
    ap.add_argument("--video", help=f"input video (default {VIDEO_DIR}/<video_id>.mp4)")
    ap.add_argument("--json", help=f"result json (default {RESULTS_DIR}/<video_id>.json)")
    ap.add_argument("--glb", help="object mesh (default: the 'object.mesh' file next to the json)")
    ap.add_argument("--out", help=f"output video (default {OUT_DIR}/<video_id>.mp4)")
    ap.add_argument("--side-by-side", action="store_true", help="input and overlay next to each other")
    ap.add_argument("--grid", action="store_true", help="also save an 8-frame overview .jpg next to the video")
    args = ap.parse_args()

    if args.all:
        for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
            vid = os.path.basename(path)[:-5]
            out = overlay_video(os.path.join(VIDEO_DIR, f"{vid}.mp4"), path, None,
                                os.path.join(OUT_DIR, f"{vid}.mp4"), args.side_by_side, args.grid)
            print("wrote", out)
    elif args.video_id or args.json:
        vid = args.video_id or os.path.basename(args.json)[:-5]
        out = overlay_video(args.video or os.path.join(VIDEO_DIR, f"{vid}.mp4"),
                            args.json or os.path.join(RESULTS_DIR, f"{vid}.json"),
                            args.glb,
                            args.out or os.path.join(OUT_DIR, f"{vid}.mp4"),
                            args.side_by_side, args.grid)
        print("wrote", out)
    else:
        ap.error("give --video_id, --json or --all")


if __name__ == "__main__":
    main()
