import argparse
import json
import os
import random
from pathlib import Path

import lightning as pl
import torch
import torchvision
from datasets import Dataset as HFDataset, DownloadConfig, concatenate_datasets, load_dataset
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
from torchvision import transforms

from diffsynth import ModelManager, SDImagePipeline
from diffsynth.models.utils import load_state_dict

os.environ["TOKENIZERS_PARALLELISM"] = "True"


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
        "cache_dir": cache_dir,
    }
    if cache_dir is not None:
        load_kwargs["download_config"] = DownloadConfig(local_files_only=True)
    return load_dataset(**load_kwargs)

try:
    from structured_noise import generate_structured_noise_batch_vectorized
except ImportError as exc:
    raise ImportError(
        "structured-noise is required. Install it with `pip install git+https://github.com/zengxianyu/structured-noise`."
    ) from exc


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
        steps_per_epoch=None,
        height=512,
        width=512,
        center_crop=True,
        random_flip=False,
        max_train_samples=None,
        sample_randomly=False,
        seed=None,
    ):
        self.dataset = load_cached_or_remote_dataset(
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            split=split,
            cache_dir=cache_dir,
        )
        if max_train_samples is not None:
            self.dataset = self.dataset.shuffle(seed=seed).select(
                range(min(max_train_samples, len(self.dataset)))
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
        self.steps_per_epoch = steps_per_epoch
        self.sample_randomly = sample_randomly
        self.seed = seed

        self.image_processor = transforms.Compose(
            [
                transforms.CenterCrop((height, width))
                if center_crop
                else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip()
                if random_flip
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.height = height
        self.width = width

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
        if self.sample_randomly:
            if self.seed is None:
                data_id = random.randrange(len(self.dataset))
            else:
                rng = random.Random(self.seed + index)
                data_id = rng.randrange(len(self.dataset))
        else:
            data_id = index % len(self.dataset)

        example = self.dataset[data_id]
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
        if self.steps_per_epoch is None:
            return len(self.dataset)
        return self.steps_per_epoch


def tensor_to_pil(image_tensor):
    image_tensor = image_tensor.detach().cpu().clamp(-1, 1)
    image_tensor = (image_tensor + 1.0) / 2.0
    image_tensor = image_tensor.permute(1, 2, 0)
    image_tensor = (image_tensor * 255.0).round().to(torch.uint8).numpy()
    return Image.fromarray(image_tensor)


class ProgressLogCallback(pl.Callback):
    def __init__(self, progress_log_steps=10):
        super().__init__()
        self.progress_log_steps = progress_log_steps

    def on_train_start(self, trainer, pl_module):
        print(
            f"Training started: epochs={trainer.max_epochs}, max_steps={trainer.max_steps}, batches_per_epoch={trainer.num_training_batches}",
            flush=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.progress_log_steps <= 0:
            return
        if trainer.global_step == 0:
            return
        if trainer.global_step % self.progress_log_steps != 0:
            return

        loss_value = None
        if torch.is_tensor(outputs):
            loss_value = float(outputs.detach().cpu())
        elif isinstance(outputs, dict):
            loss_tensor = outputs.get("loss")
            if torch.is_tensor(loss_tensor):
                loss_value = float(loss_tensor.detach().cpu())

        loss_text = f", loss={loss_value:.6f}" if loss_value is not None else ""
        print(
            f"Progress: epoch={trainer.current_epoch + 1}/{trainer.max_epochs}, step={trainer.global_step}, batch={batch_idx + 1}/{trainer.num_training_batches}{loss_text}",
            flush=True,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        print(
            f"Epoch finished: epoch={trainer.current_epoch + 1}/{trainer.max_epochs}, global_step={trainer.global_step}",
            flush=True,
        )


class ValidationPreviewCallback(pl.Callback):
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
        validation_steps=0,
        validation_epochs=0,
        num_validation_images=1,
        validation_inference_steps=30,
        validation_cfg_scale=7.5,
        validation_dir_name="validation",
        structured_noise_sampling_method="two-gaussian",
        structured_noise_transition_width=2.0,
        seed=None,
    ):
        super().__init__()
        self.validation_steps = validation_steps
        self.validation_epochs = validation_epochs
        self.num_validation_images = num_validation_images
        self.validation_inference_steps = validation_inference_steps
        self.validation_cfg_scale = validation_cfg_scale
        self.validation_dir_name = validation_dir_name
        self.structured_noise_sampling_method = structured_noise_sampling_method
        self.structured_noise_transition_width = structured_noise_transition_width
        self.seed = seed
        self.fixed_validation_indices = None
        self.dataset = SinglePromptPairDataset(
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            split=split,
            cache_dir=cache_dir,
            image_column=image_column,
            cond_image_column=cond_image_column,
            prompt=prompt,
            steps_per_epoch=None,
            height=height,
            width=width,
            center_crop=True,
            random_flip=False,
            max_train_samples=None,
            sample_randomly=False,
            seed=seed,
        )

    def _fixed_indices_path(self, trainer):
        return Path(trainer.default_root_dir) / self.validation_dir_name / "fixed_indices.json"

    def _get_fixed_validation_indices(self, trainer):
        if self.fixed_validation_indices is not None:
            return self.fixed_validation_indices

        manifest_path = self._fixed_indices_path(trainer)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.fixed_validation_indices = payload["indices"]
            return self.fixed_validation_indices

        sample_count = min(self.num_validation_images, len(self.dataset))
        if sample_count <= 0:
            self.fixed_validation_indices = []
            return self.fixed_validation_indices

        if sample_count == len(self.dataset):
            indices = list(range(len(self.dataset)))
        else:
            rng = random.Random(self.seed)
            indices = rng.sample(range(len(self.dataset)), sample_count)

        payload = {
            "dataset_length": len(self.dataset),
            "num_validation_images": sample_count,
            "indices": indices,
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.fixed_validation_indices = indices
        print(f"Saved fixed validation indices to {manifest_path}")
        return self.fixed_validation_indices

    def _save_validation_triplets(self, trainer, pl_module, force=False):
        if not force and self.validation_steps <= 0 and self.validation_epochs <= 0:
            return
        output_dir = Path(trainer.default_root_dir) / self.validation_dir_name / f"step-{trainer.global_step:06d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving validation previews to {output_dir}")

        was_training = pl_module.pipe.denoising_model().training
        original_timesteps = pl_module.pipe.scheduler.timesteps.clone()
        pl_module.pipe.eval()
        pl_module.pipe.denoising_model().eval()

        try:
            validation_indices = self._get_fixed_validation_indices(trainer)
            for sample_index, dataset_index in enumerate(validation_indices):
                sample = self.dataset[dataset_index]
                cond_tensor = sample["cond_image"]
                target_tensor = sample["image"]
                prompt = sample["text"]

                cond_pil = tensor_to_pil(cond_tensor)
                target_pil = tensor_to_pil(target_tensor)

                cond_input = pl_module.pipe.preprocess_image(cond_pil)
                cond_input = cond_input.to(dtype=pl_module.pipe.torch_dtype, device=pl_module.device)
                cond_latents = pl_module.pipe.vae_encoder(cond_input)
                noise, radius = pl_module._build_noise(cond_latents)
                use_autocast = (
                    pl_module.device.type != "cpu"
                    and pl_module.pipe.torch_dtype in (torch.float16, torch.bfloat16)
                )

                with torch.no_grad(), torch.autocast(
                    device_type=pl_module.device.type,
                    dtype=pl_module.pipe.torch_dtype,
                    enabled=use_autocast,
                ):
                    generated = pl_module.pipe(
                        prompt=prompt,
                        negative_prompt="worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw, game, rendering, cartoon, 3D",
                        cfg_scale=self.validation_cfg_scale,
                        clip_skip=1,
                        height=cond_pil.height,
                        width=cond_pil.width,
                        num_inference_steps=self.validation_inference_steps,
                        noise=noise,
                        input_image=cond_pil,
                        denoising_strength=1.0,
                    )

                triplet = Image.new("RGB", (cond_pil.width * 3, cond_pil.height))
                triplet.paste(cond_pil, (0, 0))
                triplet.paste(generated, (cond_pil.width, 0))
                triplet.paste(target_pil, (cond_pil.width * 2, 0))
                suffix = f"sample-{sample_index:02d}-idx-{dataset_index:06d}-r-{int(radius) if radius is not None else 0}.png"
                triplet.save(output_dir / suffix)
        finally:
            pl_module.pipe.scheduler.timesteps = original_timesteps
            pl_module.pipe.train(False)
            if was_training:
                pl_module.pipe.denoising_model().train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.validation_steps <= 0:
            return
        if trainer.global_step == 0:
            return
        if trainer.global_step % self.validation_steps != 0:
            return
        self._save_validation_triplets(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.validation_epochs <= 0:
            return
        current_epoch = trainer.current_epoch + 1
        if current_epoch % self.validation_epochs != 0:
            return
        self._save_validation_triplets(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self._save_validation_triplets(trainer, pl_module, force=True)


class LightningModelForSD15PPDLoRA(pl.LightningModule):
    def __init__(
        self,
        pretrained_path,
        torch_dtype=torch.float16,
        learning_rate=1e-4,
        use_gradient_checkpointing=True,
        training_mode="lora",
        lora_rank=4,
        lora_alpha=4,
        lora_target_modules="to_q,to_k,to_v,to_out",
        init_lora_weights="gaussian",
        pretrained_lora_path=None,
        structural_noise_radius=100.0,
        sample_radius_exponential=False,
        min_structural_noise_radius=10.0,
        structural_noise_lambda=0.1,
        structured_noise_transition_width=2.0,
        structured_noise_sampling_method="two-gaussian",
        seed=None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.training_mode = training_mode
        self.structural_noise_radius = structural_noise_radius
        self.sample_radius_exponential = sample_radius_exponential
        self.min_structural_noise_radius = min_structural_noise_radius
        self.structural_noise_lambda = structural_noise_lambda
        self.structured_noise_transition_width = structured_noise_transition_width
        self.structured_noise_sampling_method = structured_noise_sampling_method
        self.seed = seed
        self.lora_alpha = lora_alpha

        model_manager = ModelManager(torch_dtype=torch_dtype, device=self.device)
        model_manager.load_models([pretrained_path])
        self.pipe = SDImagePipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000)

        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

        if training_mode not in {"lora", "full"}:
            raise ValueError(f"Unsupported training mode: {training_mode}")

        if training_mode == "full" and pretrained_lora_path is not None:
            raise ValueError("--pretrained_lora_path can only be used with --training_mode lora.")

        if training_mode == "lora":
            if init_lora_weights == "kaiming":
                init_lora_weights = True
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights=init_lora_weights,
                target_modules=lora_target_modules.split(","),
            )
            inject_adapter_in_model(lora_config, self.pipe.denoising_model())
            for param in self.pipe.denoising_model().parameters():
                if param.requires_grad:
                    param.data = param.to(torch.float32)

            if pretrained_lora_path is not None:
                state_dict = load_state_dict(pretrained_lora_path)
                missing_keys, unexpected_keys = self.pipe.denoising_model().load_state_dict(
                    state_dict, strict=False
                )
                all_keys = [name for name, _ in self.pipe.denoising_model().named_parameters()]
                num_updated_keys = len(all_keys) - len(missing_keys)
                print(
                    f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {len(unexpected_keys)} parameters are unexpected."
                )
        else:
            self.pipe.denoising_model().requires_grad_(True)
            for param in self.pipe.denoising_model().parameters():
                if param.requires_grad:
                    param.data = param.to(torch.float32)

    def _sample_radius(self):
        if self.sample_radius_exponential:
            distribution = torch.distributions.Exponential(
                rate=torch.tensor(float(self.structural_noise_lambda))
            )
            sampled = distribution.sample().item()
            return self.min_structural_noise_radius + sampled
        return self.structural_noise_radius

    def _build_noise(self, cond_latents):
        radius = self._sample_radius()
        if radius is None or radius <= 0:
            noise = torch.randn_like(cond_latents)
        else:
            base_noise = torch.randn_like(cond_latents)
            noise = generate_structured_noise_batch_vectorized(
                cond_latents,
                cutoff_radius=radius,
                transition_width=self.structured_noise_transition_width,
                input_noise=base_noise,
                sampling_method=self.structured_noise_sampling_method,
            ).to(device=cond_latents.device, dtype=cond_latents.dtype)
        return noise, radius

    def training_step(self, batch, batch_idx):
        text = batch["text"]
        image = batch["image"]
        cond_image = batch["cond_image"]

        if not isinstance(text, str):
            text = text[0]

        self.pipe.device = self.device
        prompt_emb = self.pipe.encode_prompt(text, positive=True)
        latents = self.pipe.vae_encoder(
            image.to(dtype=self.pipe.torch_dtype, device=self.device)
        )
        cond_latents = self.pipe.vae_encoder(
            cond_image.to(dtype=self.pipe.torch_dtype, device=self.device)
        )
        noise, radius = self._build_noise(cond_latents)

        timestep = torch.randint(
            0,
            self.pipe.scheduler.num_train_timesteps,
            (1,),
            device=self.device,
            dtype=torch.long,
        ).to(dtype=torch.float32)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        noise_pred = self.pipe.denoising_model()(
            noisy_latents,
            timestep=timestep,
            **prompt_emb,
            **extra_input,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
        )
        loss = torch.nn.functional.mse_loss(
            noise_pred.float(), training_target.float()
        )
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        self.log("train_loss", loss, prog_bar=True)
        if radius is not None:
            self.log("train_radius", float(radius), prog_bar=False)
        return loss

    def configure_optimizers(self):
        trainable_modules = filter(
            lambda param: param.requires_grad,
            self.pipe.denoising_model().parameters(),
        )
        return torch.optim.AdamW(trainable_modules, lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        if self.training_mode == "full":
            checkpoint.update(self.pipe.denoising_model().state_dict())
            return

        trainable_param_names = list(
            filter(
                lambda named_param: named_param[1].requires_grad,
                self.pipe.denoising_model().named_parameters(),
            )
        )
        trainable_param_names = {name for name, _ in trainable_param_names}
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SD1.5 with PPD-style structured noise on a paired HF dataset."
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to the SD1.5 checkpoint, for example models/stable_diffusion/v1-5-pruned-emaonly.safetensors.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HF dataset name or local datasets-compatible path.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Optional dataset config name.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Dataset cache directory.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="edited_image",
        help="Target image column.",
    )
    parser.add_argument(
        "--cond_image_column",
        type=str,
        default="input_image",
        help="Source image column used to construct structured noise.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="make this image photorealistic",
        help="Single prompt reused for the full dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./models/train/SD1.5-PPD-filter-lora",
        help="Path to save checkpoints and logs.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help="Optional override for epoch length in samples. By default one epoch means one full pass over the dataset.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum optimizer steps. Use -1 to disable.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=0,
        help="Save a checkpoint every N training steps. If 0, saves every epoch.",
    )
    parser.add_argument(
        "--checkpointing_epochs",
        type=int,
        default=1,
        help="Save a checkpoint every N epochs when --checkpointing_steps=0.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Training image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Training image width.",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        default=False,
        help="Use center crop instead of random crop.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        default=False,
        help="Use random horizontal flip.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per device.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16", "16-mixed", "bf16"],
        help="Lightning precision mode.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Training mode. Use 'full' to fine-tune the whole SD1.5 denoiser instead of LoRA adapters.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=1,
        help="LoRA rank. Used only when --training_mode=lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=1.0,
        help="LoRA alpha. Used only when --training_mode=lora.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="to_q,to_k,to_v,to_out",
        help="Comma-separated target modules for LoRA. Used only when --training_mode=lora.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="LoRA init method. Used only when --training_mode=lora.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Optional path to resume from an existing LoRA checkpoint. Used only when --training_mode=lora.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Gradient accumulation factor.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=[
            "auto",
            "deepspeed_stage_1",
            "deepspeed_stage_2",
            "deepspeed_stage_3",
        ],
        help="Lightning training strategy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional cap on dataset samples for debugging.",
    )
    parser.add_argument(
        "--structural_noise_radius",
        type=float,
        default=100.0,
        help="Fixed structured-noise cutoff radius. Set <= 0 to fall back to Gaussian noise.",
    )
    parser.add_argument(
        "--sample_radius_exponential",
        action="store_true",
        default=False,
        help="Sample radius per batch as r0 + Exp(lambda), following the paper.",
    )
    parser.add_argument(
        "--min_structural_noise_radius",
        type=float,
        default=10.0,
        help="Minimum cutoff radius r0 used with --sample_radius_exponential.",
    )
    parser.add_argument(
        "--structural_noise_lambda",
        type=float,
        default=0.1,
        help="Exponential rate lambda used with --sample_radius_exponential.",
    )
    parser.add_argument(
        "--structured_noise_transition_width",
        type=float,
        default=2.0,
        help="Smooth transition width for the frequency cutoff.",
    )
    parser.add_argument(
        "--structured_noise_sampling_method",
        type=str,
        default="two-gaussian",
        choices=["fft", "cdf", "two-gaussian"],
        help="Sampling method for structured-noise magnitudes.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=0,
        help="Save validation triplets every N train steps. Set 0 to disable periodic validation.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=0,
        help="Save validation triplets every N epochs. Set 0 to disable epoch-based validation.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of validation triplets to save per validation run.",
    )
    parser.add_argument(
        "--validation_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps for validation previews.",
    )
    parser.add_argument(
        "--validation_cfg_scale",
        type=float,
        default=7.5,
        help="CFG scale for validation previews.",
    )
    parser.add_argument(
        "--validation_dir_name",
        type=str,
        default="validation",
        help="Subdirectory under output_path used for saved validation triplets.",
    )
    parser.add_argument(
        "--progress_log_steps",
        type=int,
        default=10,
        help="Write a plain-text progress line every N optimizer steps. Set 0 to disable.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["gpu", "cpu", "auto"],
        help="Lightning accelerator.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="Lightning devices setting, for example 'auto' or '1'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    torch_dtype = torch.float32 if args.precision == "32" else torch.float16
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16

    model = LightningModelForSD15PPDLoRA(
        pretrained_path=args.pretrained_path,
        torch_dtype=torch_dtype,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        training_mode=args.training_mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        pretrained_lora_path=args.pretrained_lora_path,
        structural_noise_radius=args.structural_noise_radius,
        sample_radius_exponential=args.sample_radius_exponential,
        min_structural_noise_radius=args.min_structural_noise_radius,
        structural_noise_lambda=args.structural_noise_lambda,
        structured_noise_transition_width=args.structured_noise_transition_width,
        structured_noise_sampling_method=args.structured_noise_sampling_method,
        seed=args.seed,
    )

    dataset = SinglePromptPairDataset(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        split=args.dataset_split,
        cache_dir=args.cache_dir,
        image_column=args.image_column,
        cond_image_column=args.cond_image_column,
        prompt=args.instance_prompt,
        steps_per_epoch=args.steps_per_epoch,
        height=args.height,
        width=args.width,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        max_train_samples=args.max_train_samples,
        seed=args.seed,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )

    checkpoint_kwargs = {"save_top_k": -1, "save_last": True}
    if args.checkpointing_steps > 0:
        checkpoint_kwargs["every_n_train_steps"] = args.checkpointing_steps
        checkpoint_kwargs["every_n_epochs"] = None
    else:
        checkpoint_kwargs["every_n_epochs"] = args.checkpointing_epochs

    callbacks = [
        pl.pytorch.callbacks.ModelCheckpoint(**checkpoint_kwargs),
        ProgressLogCallback(progress_log_steps=args.progress_log_steps),
    ]
    if args.validation_steps > 0 or args.validation_epochs > 0 or args.num_validation_images > 0:
        callbacks.append(
            ValidationPreviewCallback(
                dataset_name=args.dataset_name,
                dataset_config_name=args.dataset_config_name,
                split=args.dataset_split,
                cache_dir=args.cache_dir,
                image_column=args.image_column,
                cond_image_column=args.cond_image_column,
                prompt=args.instance_prompt,
                height=args.height,
                width=args.width,
                validation_steps=args.validation_steps,
                validation_epochs=args.validation_epochs,
                num_validation_images=args.num_validation_images,
                validation_inference_steps=args.validation_inference_steps,
                validation_cfg_scale=args.validation_cfg_scale,
                validation_dir_name=args.validation_dir_name,
                structured_noise_sampling_method=args.structured_noise_sampling_method,
                structured_noise_transition_width=args.structured_noise_transition_width,
                seed=args.seed,
            )
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=args.devices if args.devices == "auto" else int(args.devices),
        precision=args.precision,
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=None,
    )
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()