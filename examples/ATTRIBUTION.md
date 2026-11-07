# Source and license of the example

Everything in `examples/` is made from one recording of the **HO-Cap** dataset. This recording is
**not** one of the course videos.

* **Dataset**: Jikai Wang, Qifan Zhang, Yu-Wei Chao, Bowen Wen, Xiaohu Guo, Yu Xiang.
  *HO-Cap: A Capture System and Dataset for 3D Reconstruction and Pose Tracking of Hand-Object
  Interaction.* NeurIPS 2025 Datasets and Benchmarks Track.
  <https://irvlutd.github.io/HOCap/> · <https://arxiv.org/abs/2406.06843>
* **License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* **Used part**: `subject_5/20231027_112303`, camera `108222250342`, frames 157–306 (150 frames),
  object `G15_3`.

Changes made by the course TAs:

| File | What it is |
|---|---|
| `videos/example_01.mp4` | the colour frames of that camera and frame range, encoded as H.264 at 30 fps |
| `videos/videos.json` | frame count, fps and size of the clip (same layout as `data/videos/videos.json`) |
| `results/example_01.json`, `.glb` | HO-Cap's right-hand MANO and object annotations converted to the course format (README §6) in this camera's frame, **with errors added on purpose** (hand pose, shape and depth; object size, rotation and depth; focal length). It is an example of the file format, **not** ground truth and not a target accuracy. |
| `overlays/example_01.mp4`, `.jpg` | output of `tools/overlay.py --side-by-side --grid` on the files above |
