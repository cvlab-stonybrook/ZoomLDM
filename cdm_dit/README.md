# Code for training and evaluating CDM (Conditioning Diffusion Model)



## Training the CDM

To train the CDM, use the `train_cdm.py` script. Below is an example command to initiate training:

```bash
python train_cdm.py --data-path /path/to/data --results-dir results --model DiT-B 
```

Make sure to replace `/path/to/data` with the actual path to your dataset (same as used for training ZoomLDM).

## Evaluating the CDM

To evaluate the CDM, use the `cdm_measure_fid.py` script. Below is an example command to measure FID:

```bash
python cdm_measure_fid.py --cdm_ckpt_path /path/to/cdm_checkpoint --ldm_path /path/to/zoomldm_checkpoint_folder --fid_stats_path /path/to/fid_stats.npz --logdir fid_results --magnification 20x
```