import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchvision
from datasets import Dataset as HFDataset, DownloadConfig, concatenate_datasets, load_dataset
from PIL import Image
from torchvision import transforms

from diffsynth import ModelManager, SDImagePipeline
from diffsynth.models.utils import load_state_dict

try:
    from structured_noise import generate_structured_noise_batch_vectorized
except ImportError as exc:
    raise ImportError(
        "structured-noise is required. Install it with `pip install git+https://github.com/zengxianyu/structured-noise`."
    ) from exc


NUM_RANDOM_INDICES = 20

PRETRAINED_PATH = REPO_ROOT / "models/stable_diffusion/v1-5-pruned-emaonly.safetensors"
CHECKPOINT_PATH = REPO_ROOT / "models/train/SD1.5-PPD-filter-full/lightning_logs/version_6/checkpoints/epoch=9-step=61780.ckpt"
OUTPUT_DIR = CHECKPOINT_PATH.parent / "inference_selected_indices"

DATASET_NAME = "dim/nfs_pix2pix_1920_1080_v6_2x_flux_klein_4B_lora"
DATASET_CONFIG_NAME = None
DATASET_SPLIT = "train"
CACHE_DIR = Path("/code/dataset/nfs_pix2pix_1920_1080_v6_2x_flux_klein_4B_lora")
IMAGE_COLUMN = "edited_image"
COND_IMAGE_COLUMN = "input_image"
PROMPT = "make this image photorealistic"

HEIGHT = 512
WIDTH = 512
CENTER_CROP = True
RANDOM_FLIP = False

DEVICE = "cuda"
PRECISION = "32"
NUM_INFERENCE_STEPS = 30
CFG_SCALE = 1.0
CLIP_SKIP = 1
NEGATIVE_PROMPT = None

STRICT_CHECKPOINT_LOADING = True
SAMPLE_RADIUS_EXPONENTIAL = True
FIXED_STRUCTURAL_NOISE_RADIUS = 100.0
MIN_STRUCTURAL_NOISE_RADIUS = 100.0
STRUCTURAL_NOISE_LAMBDA = 0.1
STRUCTURED_NOISE_TRANSITION_WIDTH = 2.0
STRUCTURED_NOISE_SAMPLING_METHOD = "two-gaussian"
SEED = 42


def load_cached_arrow_dataset(split="train", cache_dir=None):
    if cache_dir is None:
        return None

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return None

    arrow_files = sorted(cache_path.rglob(f"*-{split}-*.arrow"))
    if not arrow_files:
        return None

    print(f"Loading dataset split '{split}' from local Arrow cache in {cache_path}")
    shard_datasets = [HFDataset.from_file(str(arrow_file)) for arrow_file in arrow_files]
    if len(shard_datasets) == 1:
        return shard_datasets[0]
    return concatenate_datasets(shard_datasets)


def load_cached_or_remote_dataset(dataset_name, dataset_config_name=None, split="train", cache_dir=None):
    cached_dataset = load_cached_arrow_dataset(split=split, cache_dir=cache_dir)
    if cached_dataset is not None:
        return cached_dataset

    load_kwargs = {
        "path": dataset_name,
        "name": dataset_config_name,
        "split": split,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
    }
    if cache_dir is not None:
        load_kwargs["download_config"] = DownloadConfig(local_files_only=True)
    return load_dataset(**load_kwargs)


class SinglePromptPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name,
        dataset_config_name=None,
        split="train",
        cache_dir=None,
        image_column="edited_image",
        cond_image_column="input_image",
        prompt="make this image photorealistic",
        height=512,
        width=512,
        center_crop=True,
        random_flip=False,
    ):
        self.dataset = load_cached_or_remote_dataset(
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            split=split,
            cache_dir=cache_dir,
        )

        column_names = set(self.dataset.column_names)
        if image_column not in column_names:
            raise ValueError(
                f"Image column '{image_column}' not found. Available columns: {sorted(column_names)}"
            )
        if cond_image_column not in column_names:
            raise ValueError(
                f"Condition column '{cond_image_column}' not found. Available columns: {sorted(column_names)}"
            )

        self.image_column = image_column
        self.cond_image_column = cond_image_column
        self.prompt = prompt
        self.height = height
        self.width = width
        self.image_processor = transforms.Compose(
            [
                transforms.CenterCrop((height, width))
                if center_crop
                else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def _resize_for_crop(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        resized_shape = [round(height * scale), round(width * scale)]
        return torchvision.transforms.functional.resize(
            image,
            resized_shape,
            interpolation=transforms.InterpolationMode.BILINEAR,
        )

    def __getitem__(self, index):
        example = self.dataset[index]
        target_image = example[self.image_column]
        cond_image = example[self.cond_image_column]

        if not isinstance(target_image, Image.Image):
            target_image = Image.fromarray(target_image)
        if not isinstance(cond_image, Image.Image):
            cond_image = Image.fromarray(cond_image)

        target_image = self._resize_for_crop(target_image.convert("RGB"))
        cond_image = self._resize_for_crop(cond_image.convert("RGB"))

        return {
            "text": self.prompt,
            "image": self.image_processor(target_image),
            "cond_image": self.image_processor(cond_image),
        }

    def __len__(self):
        return len(self.dataset)


def tensor_to_pil(image_tensor):
    image_tensor = image_tensor.detach().cpu().clamp(-1, 1)
    image_tensor = (image_tensor + 1.0) / 2.0
    image_tensor = image_tensor.permute(1, 2, 0)
    image_tensor = (image_tensor * 255.0).round().to(torch.uint8).numpy()
    return Image.fromarray(image_tensor)


def normalize_unet_state_dict(state_dict):
    normalized_state_dict = {}
    prefixes = (
        "pipe.denoising_model.",
        "pipe.unet.",
        "denoising_model.",
        "unet.",
        "model.",
    )
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        normalized_key = key
        for prefix in prefixes:
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]
                break
        normalized_state_dict[normalized_key] = value
    return normalized_state_dict


def resolve_dtype(precision):
    if precision == "32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    return torch.float16


def resolve_indices(dataset_length):
    sample_count = min(NUM_RANDOM_INDICES, dataset_length)
    if sample_count <= 0:
        raise ValueError("Dataset is empty, no indices to sample.")

    rng = random.Random(SEED) if SEED is not None else random
    return sorted(rng.sample(range(dataset_length), sample_count))


def sample_radius():
    if SAMPLE_RADIUS_EXPONENTIAL:
        distribution = torch.distributions.Exponential(rate=torch.tensor(float(STRUCTURAL_NOISE_LAMBDA)))
        return MIN_STRUCTURAL_NOISE_RADIUS + distribution.sample().item()
    return FIXED_STRUCTURAL_NOISE_RADIUS


def build_noise(cond_latents):
    radius = sample_radius()
    if radius is None or radius <= 0:
        return torch.randn_like(cond_latents), radius

    base_noise = torch.randn_like(cond_latents)
    noise = generate_structured_noise_batch_vectorized(
        cond_latents,
        cutoff_radius=radius,
        transition_width=STRUCTURED_NOISE_TRANSITION_WIDTH,
        input_noise=base_noise,
        sampling_method=STRUCTURED_NOISE_SAMPLING_METHOD,
    ).to(device=cond_latents.device, dtype=cond_latents.dtype)
    return noise, radius


def build_pipeline(device, torch_dtype):
    model_manager = ModelManager(torch_dtype=torch_dtype, device=device)
    model_manager.load_models([str(PRETRAINED_PATH)])
    pipe = SDImagePipeline.from_model_manager(model_manager)
    pipe.scheduler.set_timesteps(1000)
    pipe.requires_grad_(False)
    pipe.eval()
    pipe.device = torch.device(device)
    return pipe


def load_checkpoint_into_pipe(pipe):
    checkpoint = load_state_dict(str(CHECKPOINT_PATH))
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format in {CHECKPOINT_PATH}")

    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]
    trained_state_dict = normalize_unet_state_dict(checkpoint)

    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    missing_keys, unexpected_keys = pipe.denoising_model().load_state_dict(
        trained_state_dict,
        strict=STRICT_CHECKPOINT_LOADING,
    )
    if not STRICT_CHECKPOINT_LOADING:
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")


def main():
    if SEED is not None:
        random.seed(SEED)
        torch.manual_seed(SEED)

    torch_dtype = resolve_dtype(PRECISION)
    dataset = SinglePromptPairDataset(
        dataset_name=DATASET_NAME,
        dataset_config_name=DATASET_CONFIG_NAME,
        split=DATASET_SPLIT,
        cache_dir=CACHE_DIR,
        image_column=IMAGE_COLUMN,
        cond_image_column=COND_IMAGE_COLUMN,
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        center_crop=CENTER_CROP,
        random_flip=RANDOM_FLIP,
    )

    indices = resolve_indices(len(dataset))
    print(f"Selected dataset indices: {indices}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline(DEVICE, torch_dtype)
    load_checkpoint_into_pipe(pipe)

    use_autocast = DEVICE != "cpu" and pipe.torch_dtype in (torch.float16, torch.bfloat16)
    device = torch.device(DEVICE)
    manifest = {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "dataset_name": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "indices": indices,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "cfg_scale": CFG_SCALE,
        "precision": PRECISION,
        "device": DEVICE,
        "samples": [],
    }

    for sample_number, dataset_index in enumerate(indices):
        sample = dataset[dataset_index]
        cond_pil = tensor_to_pil(sample["cond_image"])
        target_pil = tensor_to_pil(sample["image"])

        cond_input = pipe.preprocess_image(cond_pil).to(dtype=pipe.torch_dtype, device=device)
        with torch.no_grad():
            cond_latents = pipe.vae_encoder(cond_input)
            noise, radius = build_noise(cond_latents)

        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=pipe.torch_dtype,
            enabled=use_autocast,
        ):
            generation_kwargs = {
                "prompt": sample["text"],
                "cfg_scale": CFG_SCALE,
                "clip_skip": CLIP_SKIP,
                "height": cond_pil.height,
                "width": cond_pil.width,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "noise": noise,
                "input_image": cond_pil,
                "denoising_strength": 1.0,
            }
            if NEGATIVE_PROMPT:
                generation_kwargs["negative_prompt"] = NEGATIVE_PROMPT

            generated = pipe(
                **generation_kwargs,
            )

        file_name = f"sample-{sample_number:02d}-idx-{dataset_index:06d}-r-{int(radius) if radius is not None else 0}.png"
        output_path = OUTPUT_DIR / file_name
        triplet = Image.new("RGB", (cond_pil.width * 3, cond_pil.height))
        triplet.paste(cond_pil, (0, 0))
        triplet.paste(generated, (cond_pil.width, 0))
        triplet.paste(target_pil, (cond_pil.width * 2, 0))
        triplet.save(output_path)
        print(f"Saved {output_path}")

        manifest["samples"].append(
            {
                "dataset_index": dataset_index,
                "radius": radius,
                "output": output_path.name,
            }
        )

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()