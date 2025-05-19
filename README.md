# ZoomLDM: Latent Diffusion Model for multi-scale image generation 

### <div align="center"> CVPR 2025 Poster <div>

<div align="center">
    <a href="https://histodiffusion.github.io/docs/projects/zoomldm/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=red&logo=github-pages"></a> &ensp;
    <a href="https://arxiv.org/abs/2411.16969"><img src="https://img.shields.io/static/v1?label=arXiv&message=ZoomLDM&color=b75fb3&logo=arxiv"></a> &ensp;
    <a href="https://huggingface.co/StonyBrook-CVLab/ZoomLDM"><img src="https://img.shields.io/static/v1?label=HF&message=Checkpoints&color=69a75b&logo=huggingface"></a> &ensp;
    <a href="https://huggingface.co/datasets/StonyBrook-CVLab/ZoomLDM-demo-dataset"><img src="https://img.shields.io/static/v1?label=HF&message=Example%20Dataset&color=6785d0&logo=huggingface"></a> &ensp;
    <a href="https://histodiffusion.github.io/pages/zoomldm_large_images/large_images.html"><img src="https://img.shields.io/badge/Large%20image%20Viewer-cc5658"></a> &ensp;
</div>

## Setup

```bash
git clone https://github.com/cvlab-stonybrook/ZoomLDM/
conda activate zoomldm
pip install -r requirements.txt
```

## Inference

Refer to these notebooks for sampling images using the pretrained model:

- [Patch level generation](./notebooks/sample_patches.ipynb)
- Large image generation : coming soon
- Super resolution : coming soon

## Training
Coming soon

## Bibtex
```bibtex
@article{yellapragada2024zoomldm,
  title={ZoomLDM: Latent Diffusion Model for multi-scale image generation},
  author={Yellapragada, Srikar and Graikos, Alexandros and Triaridis, Kostas and Prasanna, Prateek and Gupta, Rajarsi R and Saltz, Joel and Samaras, Dimitris},
  journal={arXiv preprint arXiv:2411.16969},
  year={2024}
}
```