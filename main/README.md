# main/

Your pipeline. Requirements:

* **One entry point** that runs every video in `data/` and writes `results/<video_id>.json` and
  `results/<video_id>.glb` [TBD: required name, e.g. `python main/run.py --config main/config.yaml`].
* **One configuration** for all videos: no per-video code paths or hand-tuned parameters.
* Reproducible: the TAs re-run it on a sample of submissions with only `data/`, `mano_models/` and
  public model weights. Fix random seeds where you can.
