"""Render your hand and object meshes from three views and save a video (mp4). Example: README §5.3.

    python tools/mesh_view.py --video_id easy_01   # data/easy_01/frames + results/easy_01.json -> mesh_views/easy_01.mp4
    python tools/mesh_view.py --all                # every results/*.json

Panels: input | camera | side (from the right) | top. Only the front of each face is drawn, so holes
and flipped faces appear as missing surface. Needs mano_models/MANO_RIGHT.pkl (README §2).
"""
import argparse
import glob
import os

import imageio
import numpy as np
import smplx
import trimesh
from PIL import Image, ImageDraw

from overlay import DATA_DIR, HAND_RGB, OBJ_RGB, RESULTS_DIR, load_result, save_grid

OUT_DIR, MANO_DIR = "mesh_views", "mano_models"
BG_RGB = (35, 35, 40)
LIGHT = np.array([-0.4, -0.6, -1.0])            # from the upper left, behind the viewer
LIGHT /= np.linalg.norm(LIGHT)
MIN_EXTENT = 0.25                               # m; the views never zoom in closer than this


def look_at(forward, up):
    """View rotation; rows = image right, image down, viewing direction (camera coordinates)."""
    forward = np.asarray(forward, np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(forward, right), forward])


VIEWS = {"camera": look_at([0, 0, 1], [0, -1, 0]),      # as the input camera sees it
         "side": look_at([-1, 0, 0], [0, -1, 0]),       # from the right: depth runs left to right
         "top": look_at([0, 1, 0], [0, 0, 1])}          # from above: the camera is at the bottom


def render(meshes, R, centre, scale, size, ss=2):
    """Orthographic view of [(verts, faces, rgb)]: front faces painted far to near, shaded by the light."""
    polys, depths, colours = [], [], []
    for verts, faces, rgb in meshes:
        tri = ((verts - centre) @ R.T)[faces]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
        front = normal[:, 2] < 0                                    # the viewer looks along +z
        shade = 0.25 + 0.75 * np.clip(normal[front] @ LIGHT, 0, 1)
        polys.append((tri[front, :, :2] * scale + size / 2) * ss)
        depths.append(tri[front, :, 2].mean(1))
        colours.append((np.asarray(rgb) * shade[:, None]).astype(int))
    polys, depths, colours = np.concatenate(polys), np.concatenate(depths), np.concatenate(colours)

    img = Image.new("RGB", (size * ss, size * ss), BG_RGB)          # drawn ss times larger, then reduced
    draw = ImageDraw.Draw(img)
    order = np.argsort(-depths)
    for poly, colour in zip(polys[order].reshape(-1, 6).tolist(), colours[order].tolist()):
        draw.polygon(poly, fill=tuple(colour))
    return img.resize((size, size), Image.LANCZOS)


def smooth(x, n=15):
    """Moving average over the frames of (F,C), edges repeated."""
    x = np.pad(x, ((n // 2, n - 1 - n // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(c, np.ones(n) / n, "valid") for c in x.T], axis=1)


def framing(r):
    """Per frame: centre (F,3) and size (F,) of the box around the hand and the object, smoothed over time."""
    obj = r["obj_verts"] @ r["R"].transpose(0, 2, 1) + r["t"][:, None]
    pts = np.concatenate([obj, r["hand_verts"]], axis=1)
    lo, hi = pts.min(1), pts.max(1)
    extent = np.maximum((hi - lo).max(1), MIN_EXTENT)
    return smooth((lo + hi) / 2), smooth(extent[:, None])[:, 0]


def mesh_summary(r):
    mesh = trimesh.Trimesh(r["obj_verts"], r["obj_faces"])          # merges vertices split along texture seams
    size = np.sort(mesh.bounding_box_oriented.extents)[::-1] * 100
    return (f"{r['video_id']} object: {len(r['obj_verts'])} vertices, {len(r['obj_faces'])} faces, "
            f"{' x '.join(f'{s:.1f}' for s in size)} cm, {mesh.body_count} part(s), "
            f"{'closed' if mesh.is_watertight else 'not closed (holes)'}")


def draw_frame(src, r, t, hand_faces, centre, extent, size):
    """The input frame and the three views, side by side."""
    src = src.resize((2 * round(src.width * size / src.height / 2), size))      # even width for H.264
    meshes = [(r["obj_verts"] @ r["R"][t].T + r["t"][t], r["obj_faces"], OBJ_RGB),
              (r["hand_verts"][t], hand_faces, HAND_RGB)]
    panels = {"input": src}
    for name, R in VIEWS.items():
        panels[name] = render(meshes, R, centre, 0.85 * size / extent, size)

    img = Image.new("RGB", (sum(p.width for p in panels.values()), size))
    draw, x = ImageDraw.Draw(img), 0
    for name, panel in panels.items():
        img.paste(panel, (x, 0))
        draw.text((x + 6, 4), f"{r['video_id']}  t={t}" if name == "input" else name, fill=(255, 255, 255))
        x += panel.width
    return img


def mesh_video(frames_dir, json_path, glb_path, out, hand_faces, fps=30, size=480, grid=False):
    r = load_result(json_path, glb_path)
    if len(glob.glob(os.path.join(frames_dir, "*.jpg"))) != r["F"]:
        raise SystemExit(f"{frames_dir} does not have num_frames = {r['F']} frames (README §5)")
    print(mesh_summary(r))
    centre, extent = framing(r)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    writer = imageio.get_writer(out, "ffmpeg", fps=fps, codec="libx264", quality=None, pixelformat="yuv420p",
                                macro_block_size=2, output_params=["-crf", "20"])
    grid_frames = set(np.linspace(0, r["F"] - 1, min(8, r["F"])).round().astype(int))
    tiles = []
    try:
        for t in range(r["F"]):
            src = Image.open(os.path.join(frames_dir, f"{t:06d}.jpg")).convert("RGB")
            img = draw_frame(src, r, t, hand_faces, centre[t], extent[t], size)
            if grid and t in grid_frames:
                tiles.append(img)
            writer.append_data(np.asarray(img))
    finally:
        writer.close()
    if tiles:
        save_grid(tiles, os.path.splitext(out)[0] + ".jpg", width=1040, cols=2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video_id", help="use the default paths for this video id")
    ap.add_argument("--all", action="store_true", help=f"every {RESULTS_DIR}/*.json")
    ap.add_argument("--frames", help=f"input frames folder (default {DATA_DIR}/<video_id>/frames)")
    ap.add_argument("--json", help=f"result json (default {RESULTS_DIR}/<video_id>.json)")
    ap.add_argument("--glb", help="object mesh (default: the 'object.mesh' file next to the json)")
    ap.add_argument("--out", help=f"output video (default {OUT_DIR}/<video_id>.mp4)")
    ap.add_argument("--mano", default=MANO_DIR, help=f"folder with MANO_RIGHT.pkl (default {MANO_DIR})")
    ap.add_argument("--fps", type=float, default=30, help="frame rate of the output video (default 30)")
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
    hand_faces = np.asarray(smplx.MANO(args.mano, is_rhand=True).faces, np.int64)
    for frames_dir, json_path, glb_path, out in jobs:
        print("wrote", mesh_video(frames_dir, json_path, glb_path, out, hand_faces, args.fps, grid=args.grid))


if __name__ == "__main__":
    main()
