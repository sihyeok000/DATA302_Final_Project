# submodules/

External models used by your pipeline, added as git submodules so the exact version is recorded:

```bash
git submodule add <repository-url> submodules/<name>
git -C submodules/<name> checkout <commit>      # pin the version you used
```

Models used in the course slides [TBD: repository URLs and recommended commits]:

| Stage | Model |
|---|---|
| masks (hand, object) | SAM3 |
| depth + intrinsics | MoGe-3 |
| hand trajectory | HaPTIC |
| object mesh | SAM 3D Objects |
| object pose | FoundationPose |
| reference for optimization | EasyHOI |

Model weights and checkpoints are not committed; document in your report where they come from.
