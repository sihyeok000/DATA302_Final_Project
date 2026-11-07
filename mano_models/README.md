# mano_models/

Put the official MANO right-hand model here:

```
mano_models/MANO_RIGHT.pkl
```

1. Register at <https://mano.is.tue.mpg.de> and download **Models & Code**.
2. Copy `MANO_RIGHT.pkl` from the archive into this folder.

The MANO license does not allow redistribution: do **not** commit the file or put it in your
submission (`.gitignore` excludes `*.pkl` here). Reading it needs `smplx` and the GitHub version of
`chumpy`.

Course settings: `smplx.MANO("mano_models", is_rhand=True, use_pca=False, flat_hand_mean=True)`.
