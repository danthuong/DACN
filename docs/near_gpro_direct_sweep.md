# Near-GPro direct replacement sweep

This sweep compares two depth formulas on the exact same forward pass:

- `current`: `(Z_query + Z_size + valid_or_Zmap(Z_gpro)) / 3`
- `direct`: `Z_gpro_center` when replacement geometry is valid, otherwise the
  original MonoDETR final depth.

Queries, boxes, classes, dimensions, orientations, scores, and predicted depth
uncertainty are held fixed. Therefore `AP_direct - AP_current` isolates the
effect of changing scalar object depth.

## Full checkpoint sweep

```bash
CUDA_VISIBLE_DEVICES=1 \
python3 -u tools/sweep_near_gpro_direct.py \
  --config configs/near_gpro_zmap_replace_scratch.yaml \
  --env-file .env \
  2>&1 | tee logs/near_gpro_direct_sweep.log
```

By default this discovers `checkpoint_best.pth`, `checkpoint_last.pth`, and all
`checkpoint_epoch_N.pth` files. Files representing the same epoch are evaluated
only once.

To test one or several selected checkpoints first, repeat `--checkpoint`:

```bash
CUDA_VISIBLE_DEVICES=1 \
python3 -u tools/sweep_near_gpro_direct.py \
  --config configs/near_gpro_zmap_replace_scratch.yaml \
  --env-file .env \
  --checkpoint outputs/near_gpro_zmap_scratch/near_gpro_zmap_replace_scratch/checkpoint_best.pth \
  --checkpoint outputs/near_gpro_zmap_scratch/near_gpro_zmap_replace_scratch/checkpoint_epoch_60.pth
```

If checkpoints live elsewhere, pass `--checkpoint-dir PATH`.

## Outputs

The default summary directory is `CHECKPOINT_DIR/direct_sweep/`:

- `near_gpro_direct_sweep.svg`: AP, depth MAE, and depth P90 curves.
- `near_gpro_direct_sweep.md`: readable summary and checkpoint table.
- `near_gpro_direct_sweep.csv`: one row per checkpoint.
- `near_gpro_direct_sweep.json`: machine-readable results and protocol.
- `per_checkpoint/epoch_N/{current,direct}/`: full depth reports.

KITTI predictions and inference artifacts are retained below
`CHECKPOINT_DIR/outputs/direct_sweep/` so every reported number is auditable.
The summary is refreshed after each checkpoint, so completed results survive an
interrupted long sweep.

`GPro MAE` is valid-only. `Direct final MAE` also includes invalid-geometry
objects via the MonoDETR fallback and is therefore the fair end-to-end number.
