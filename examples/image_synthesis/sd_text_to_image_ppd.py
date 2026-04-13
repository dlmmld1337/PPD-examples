from diffsynth import ModelManager, SDImagePipeline, download_models
from PIL import Image
from structured_noise import generate_structured_noise_batch_vectorized
from peft import LoraConfig, inject_adapter_in_model
import torch
from diffsynth.models.utils import load_state_dict
import argparse


def _normalize_unet_state_dict(state_dict):
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
                normalized_key = normalized_key[len(prefix):]
                break
        normalized_state_dict[normalized_key] = value
    return normalized_state_dict


def load_trained_model_from_checkpoint(checkpoint_path, pipe, use_ema=True, strict_loading=True):
    if checkpoint_path is None:
        print("No checkpoint path provided, only loading base model")
        return pipe
    
    # Load the checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")
    
    # Load weights based on use_ema parameter
    if use_ema and 'ema_state_dict' in checkpoint:
        print("Loading EMA weights from checkpoint...")
        trained_state_dict = _normalize_unet_state_dict(checkpoint['ema_state_dict'])
        print(f"Found {len(trained_state_dict)} EMA parameters in checkpoint")
    else:
        if use_ema and 'ema_state_dict' not in checkpoint:
            print("EMA weights not found in checkpoint, falling back to regular weights.")
        if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            trained_state_dict = _normalize_unet_state_dict(checkpoint['state_dict'])
        else:
            trained_state_dict = _normalize_unet_state_dict(checkpoint)
        print("Loading regular trained parameters from checkpoint...")
        print(f"Found {len(trained_state_dict)} regular parameters in checkpoint")
    
    # Load the trained parameters into the UNet
    unet = pipe.unet if getattr(pipe, "unet", None) is not None else pipe.denoising_model()
    if unet is not None:
        # Validate that we have the right checkpoint for this model
        unet_state_dict = unet.state_dict()
        unet_param_names = set(unet_state_dict.keys())
        checkpoint_param_names = set(trained_state_dict.keys())
        
        # Check for missing keys in checkpoint
        missing_in_checkpoint = unet_param_names - checkpoint_param_names
        if missing_in_checkpoint:
            print(f"WARNING: {len(missing_in_checkpoint)} parameters missing in checkpoint:")
            for key in sorted(missing_in_checkpoint):
                print(f"  - {key}")
        
        # Check for unexpected keys in checkpoint
        unexpected_in_checkpoint = checkpoint_param_names - unet_param_names
        if unexpected_in_checkpoint:
            print(f"WARNING: {len(unexpected_in_checkpoint)} unexpected parameters in checkpoint:")
            for key in sorted(unexpected_in_checkpoint):
                print(f"  - {key}")
        
        # Check if we have any matching parameters
        matching_params = unet_param_names & checkpoint_param_names
        if not matching_params:
            raise ValueError("No matching parameters found between checkpoint and model! This checkpoint may be incompatible.")
        
        print(f"Found {len(matching_params)} matching parameters out of {len(unet_param_names)} total model parameters")
        
        # Load with validation (strict by default for safety)
        if strict_loading:
            print("Using strict loading (default) - will fail on any parameter mismatch")
            unet.load_state_dict(trained_state_dict, strict=True)
            print("All parameters loaded successfully with strict=True")
            missing_keys = []
            unexpected_keys = []
        else:
            print("Using lenient loading - allowing parameter mismatches with warnings")
            missing_keys, unexpected_keys = unet.load_state_dict(trained_state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys (will use original model weights): {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys (ignored): {len(unexpected_keys)}")
        
        if use_ema and 'ema_state_dict' in checkpoint:
            print("Successfully loaded EMA weights into UNet")
        else:
            print("Successfully loaded regular trained parameters into UNet")
            
        # Final validation - check if any critical parameters are missing
        if missing_keys:
            critical_missing = [key for key in missing_keys if any(critical in key.lower() for critical in ['conv', 'linear', 'norm', 'attention'])]
            if critical_missing:
                print(f"WARNING: Critical parameters missing: {critical_missing}")
                print("The model may not work as expected!")
            
    else:
        raise ValueError("Could not find UNet model in base model")
    
    return pipe


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with trained model (EMA or regular weights)")
    parser.add_argument(
        "--lora_checkpoint_path",
        type=str,
        required=False,
        default=None,
        help="Path to lora checkpoint file"
    )
    parser.add_argument(
        "--radius",
        type=float,
        required=False,
        default=15,
        help="Radius of structured noise"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        required=False,
        default=8,
        help="Rank of LoRA"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        required=False,
        default=4,
        help="Alpha of LoRA"
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        required=False,
        default="to_q,to_k,to_v,to_out",
        help="Target modules of LoRA"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        default="models/ppd/sd1.5-epoch=10-step=720000.ckpt",
        help="Path to the trained checkpoint file"
    )
    parser.add_argument(
        "--use_ema",
        type=bool,
        default=True,
        help="Use EMA weights (default: True)"
    )
    parser.add_argument(
        "--use_regular",
        type=bool,
        default=False,
        help="Use regular weights (overrides use_ema if both are True)"
    )
    parser.add_argument(
        "--save_intermediates",
        default=False,
        action="store_true",
        help="Save intermediate images during generation"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="output.png",
        help="Output image filename"
    )
    parser.add_argument(
        "--lenient_loading",
        action="store_true",
        help="Use lenient loading (allows parameter mismatches with warnings)"
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default="ppd/dog.jpg",
        help="Path to a single input image file (for backward compatibility)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cute border collie",
        help="Prompt for the image (used only for single image mode)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of the image"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width of the image"
    )
    parser.add_argument(
        "--partial_denoising",
        action="store_true",
        help="Use partial denoising (only denoise the structured noise)"
    )
    parser.add_argument(
        "--partial_denoising_strength",
        type=float,
        default=0.5,
        help="Strength of the partial denoising"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Execution device for model loading and inference. Use 'cpu' for smoke tests.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.lora_checkpoint_path is not None and args.checkpoint_path == "models/ppd/sd1.5-epoch=10-step=720000.ckpt":
        args.checkpoint_path = None
    height = args.height
    width = args.width

    download_models(["StableDiffusion_v15"])
    base_model_path = "models/stable_diffusion/v1-5-pruned-emaonly.safetensors"
    
    # Determine which weights to use
    use_ema = args.use_ema and not args.use_regular

    # First load the base model
    print(f"Loading base model from: {base_model_path}")
    manager_dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    model_manager = ModelManager(torch_dtype=manager_dtype, device=args.device)
    model_manager.load_models([base_model_path])
    assert (args.checkpoint_path is not None) ^ (args.lora_checkpoint_path is not None), "Either checkpoint or lora checkpoint must be provided, but not both"

    pipe = SDImagePipeline.from_model_manager(model_manager)

    # Load the trained model
    if args.checkpoint_path is not None:
        pipe = load_trained_model_from_checkpoint(args.checkpoint_path, pipe, use_ema, not args.lenient_loading)

    if args.lora_checkpoint_path is not None:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(","),
        )
        denoising_model = pipe.unet if getattr(pipe, "unet", None) is not None else pipe.denoising_model()
        pipe.unet = inject_adapter_in_model(lora_config, denoising_model)
        # Lora pretrained lora weights
        state_dict = load_state_dict(args.lora_checkpoint_path)
        missing_keys, unexpected_keys = pipe.unet.load_state_dict(state_dict, strict=False)
        all_keys = [i for i, _ in pipe.unet.named_parameters()]
        num_updated_keys = len(all_keys) - len(missing_keys)
        num_unexpected_keys = len(unexpected_keys)
        print(f"{num_updated_keys} parameters are loaded from {args.lora_checkpoint_path}. {num_unexpected_keys} parameters are unexpected.")

    image_in_pil = Image.open(args.input_image).resize((height, width), Image.Resampling.LANCZOS).convert("RGB")
    prompt = args.prompt
    negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw, game, rendering, cartoon, 3D"
    inference_steps = args.num_inference_steps

    # Generate image with the trained model
    image_in = pipe.preprocess_image(image_in_pil)
    latents = pipe.vae_encoder(image_in.to(dtype=pipe.torch_dtype, device=pipe.device))
    noise = torch.randn_like(latents)
    structured_noise = generate_structured_noise_batch_vectorized(latents, cutoff_radius=args.radius, input_noise=noise, sampling_method='two-gaussian')

    print(f"Generating image with {'EMA' if use_ema else 'regular'} trained model...")
    with torch.no_grad():
        reference_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            cfg_scale=7.5, clip_skip=1,
            height=height, width=width, num_inference_steps=inference_steps,
            noise=structured_noise,
            return_latents=args.save_intermediates,
            input_image=image_in_pil if not args.partial_denoising else None,
            denoising_strength=args.partial_denoising_strength if args.partial_denoising else 1.0,
        )
        if args.save_intermediates:
            reference_image, list_latents = reference_image
            list_intermediate = []
            for latents in [structured_noise] + list_latents:
                decoded_image = pipe.decode_image(latents.detach(), tiled=False)
                list_intermediate.append(decoded_image)
            for idx, img in enumerate(list_intermediate):
                img.save(f"intermediate/{idx:02d}.png")
                print(f"Saved intermediate image: intermediate/{idx:02d}.png")
        reference_image.save(args.output_name)
        print(f"Generated image saved as: {args.output_name}")
