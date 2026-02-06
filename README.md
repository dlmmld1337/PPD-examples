<div align="center">

# Phase-Preserving Diffusion

**Structure-Aligned Generation for Games, Videos, and Simulators**

[Paper](https://arxiv.org/abs/2512.05106) | [Project Page](https://yuzeng-at-tri.github.io/ppd-page/) | [Video](https://youtu.be/pqmdsO1L45w) | [ComfyUI](phase_preserving_flux_dev.json) | [Structured Noise](https://github.com/zengxianyu/structured-noise)

</div>

Phase-Preserving Diffusion (**PPD**) is a drop-in change to the diffusion process that preserves image phase while diffusing magnitude, enabling geometry-consistent re-rendering. It works with any diffusion model (SD 1.5, FLUX, Wan) without architectural modifications or additional parameters.

> Structural information is encoded in the phase. By replacing standard Gaussian noise with frequency-selective structured (FSS) noise, PPD preserves low-frequency phase to maintain geometry while allowing high-frequency appearance variation, controlled by a single cutoff radius parameter `r`.

## Installation

```bash
pip install -r requirements.txt
pip install git+https://github.com/zengxianyu/structured-noise
```

Download [model weights](https://huggingface.co/zengxianyu/ppd/tree/main) and place them in `models/ppd/`. Example input images are also available there.

## Inference

**SD 1.5**
```bash
PYTHONPATH=. python examples/image_synthesis/sd_text_to_image_ppd.py \
  --input_image dog.jpg \
  --radius 15 \
  --prompt "A high quality picture captured by a professional camera. Picture of a cute border collie" \
  --output output.png
```

**FLUX.1-dev**
```bash
PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_ppd.py \
  --input_image test2.jpg \
  --prompt "$(cat test2.txt)" \
  --output output.png \
  --radius 30
```

**Wan2.2-14b**
```bash
PYTHONPATH=. python examples/wanvideo/model_inference/Wan2.2-I2V-A14B_ppd.py \
  --input_image output.png \
  --input_video test2.mp4 \
  --prompt "$(cat test2.txt)" \
  --radius 30 \
  --output output.mp4
```

## Training

**FLUX**
```bash
PYTHONPATH=. bash examples/flux/model_training/lora/PPD-FLUX.1-dev.sh
```
Uses [photo-concept-bucket](https://huggingface.co/datasets/bghira/photo-concept-bucket) by default.

**Wan** &mdash; see training scripts in `examples/wanvideo/`. Uses [open-sora-pexels-subset](https://huggingface.co/datasets/zengxianyu/open-sora-pexels-subset) by default.

## Acknowledgements

This repo is largely based on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). Refer to the original repo for additional training scripts and use cases.

## Citation

```bibtex
@article{zeng2025neuralremaster,
  title   = {{NeuralRemaster}: Phase-Preserving Diffusion for Structure-Aligned Generation},
  author  = {Zeng, Yu and Ochoa, Charles and Zhou, Mingyuan and Patel, Vishal M and
             Guizilini, Vitor and McAllister, Rowan},
  journal = {arXiv preprint arXiv:2512.05106},
  year    = {2025}
}
```
