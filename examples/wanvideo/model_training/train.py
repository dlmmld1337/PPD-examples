import torch, os, json, io, tempfile
import numpy as np
import imageio
from PIL import Image
from datasets import load_dataset as hf_load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import DataLoaderConfiguration
from structured_noise import generate_structured_noise_batch_vectorized
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import diffsynth.models.wan_video_dit as _wan_dit
#_wan_dit.SAGE_ATTN_AVAILABLE = False  # SageAttention requires >64KB shared memory (Ampere+)
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import ImageCropAndResize
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _decode_video_bytes(video_bytes, num_frames, frame_processor):
    """Decode raw video bytes into a list of processed PIL Image frames."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(video_bytes)
        tmp.close()
        reader = imageio.get_reader(tmp.name)
        total = reader.count_frames()
        n = min(num_frames, total)
        # Ensure (n - 1) % 4 == 0 for Wan models
        while n > 1 and n % 4 != 1:
            n -= 1
        frames = []
        for i in range(n):
            frame = Image.fromarray(reader.get_data(i))
            frame = frame_processor(frame)
            frames.append(frame)
        reader.close()
    finally:
        os.unlink(tmp.name)
    return frames


def _process_sample(sample, captions, num_frames, frame_processor):
    """Convert a raw HuggingFace WebDataset sample into training format."""
    key = sample.get("__key__", "")
    caption = captions.get(key, "")
    video_bytes = sample["mp4"]
    if isinstance(video_bytes, dict):
        video_bytes = video_bytes["bytes"]
    frames = _decode_video_bytes(video_bytes, num_frames, frame_processor)
    return {"video": frames, "prompt": caption}


def create_hf_dataset(repo_id, num_frames=81, height=None, width=None,
                       max_pixels=1280*720, streaming=True):
    """Create a video dataset from a HuggingFace Hub repo.

    Args:
        repo_id: HuggingFace dataset repo ID.
        num_frames: Max number of video frames to load per sample.
        height: Target frame height (None for dynamic resolution).
        width: Target frame width (None for dynamic resolution).
        max_pixels: Max pixels per frame when using dynamic resolution.
        streaming: If True (default), stream data on-the-fly without full download.
                   If False, download and cache the full dataset first.

    Returns:
        HFStreamingVideoDataset (streaming=True) or HFMapVideoDataset (streaming=False).
    """
    # Load captions from annotations parquet (small, always downloaded)
    captions_ds = hf_load_dataset(repo_id, data_files="annotations/captions.parquet", split="train")
    captions = {}
    # Prefer "513f" captions (higher quality); "513f" sorts before "65f" alphabetically
    for row in captions_ds.sort("source"):
        sid = row["sample_id"]
        if sid not in captions:
            captions[sid] = row["caption"]

    # List video TAR shards in the repo
    api = HfApi()
    repo_files = api.list_repo_files(repo_id, repo_type="dataset")
    tar_files = sorted(f for f in repo_files if f.startswith("videos/") and f.endswith(".tar"))
    tar_urls = [f"https://huggingface.co/datasets/{repo_id}/resolve/main/{f}" for f in tar_files]

    # Load videos using HF WebDataset loader
    video_ds = hf_load_dataset("webdataset", data_files={"train": tar_urls},
                                split="train", streaming=streaming)

    frame_processor = ImageCropAndResize(height, width, max_pixels, 16, 16)

    if streaming:
        return HFStreamingVideoDataset(video_ds, captions, num_frames, frame_processor)
    else:
        return HFMapVideoDataset(video_ds, captions, num_frames, frame_processor)


class HFStreamingVideoDataset(torch.utils.data.IterableDataset):
    """Streaming video dataset from HuggingFace Hub (no full download required)."""

    def __init__(self, video_ds, captions, num_frames, frame_processor):
        self.video_ds = video_ds
        self.captions = captions
        self.num_frames = num_frames
        self.frame_processor = frame_processor
        self.load_from_cache = False

    def __iter__(self):
        for sample in self.video_ds:
            yield _process_sample(sample, self.captions, self.num_frames, self.frame_processor)


class HFMapVideoDataset(torch.utils.data.Dataset):
    """Map-style video dataset from HuggingFace Hub (downloads and caches first)."""

    def __init__(self, video_ds, captions, num_frames, frame_processor):
        self.video_ds = video_ds
        self.captions = captions
        self.num_frames = num_frames
        self.frame_processor = frame_processor
        self.load_from_cache = False

    def __getitem__(self, idx):
        return _process_sample(self.video_ds[idx], self.captions, self.num_frames, self.frame_processor)

    def __len__(self):
        return len(self.video_ds)


def launch_streaming_training(dataset, model, model_logger, args):
    """Training loop for streaming (iterable) datasets. Like launch_training_task but without shuffle."""
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, collate_fn=lambda x: x[0], num_workers=args.dataset_num_workers
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    # Update pipeline device references after accelerate moves params to GPU
    pipe = model.module.pipe if hasattr(model, 'module') else model.pipe
    pipe.device = accelerator.device
    pipe.vae_device = accelerator.device

    for epoch_id in range(args.num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, args.save_steps)
                scheduler.step()
        if args.save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, args.save_steps)


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(model_id=audio_processor_config.split(":")[0], origin_file_pattern=audio_processor_config.split(":")[1])
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, audio_processor_config=audio_processor_config)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary


    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]

        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}


    def forward(self, data, inputs=None):
        if inputs is None:
            with torch.no_grad():
                inputs = self.forward_preprocess(data)
        input_latents = inputs["input_latents"]
        cutoff_radius = np.random.exponential(scale=1/0.1)
        input_noise = torch.randn_like(input_latents[0].transpose(0,1).float())
        structured_noise = generate_structured_noise_batch_vectorized(input_latents[0].transpose(0,1), cutoff_radius=cutoff_radius, input_noise=input_noise)
        structured_noise = structured_noise.transpose(0,1)[None].contiguous()
        inputs["noise"] = structured_noise.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    parser.add_argument("--hf_dataset", type=str, default="zengxianyu/open-sora-pexels-subset",
                        help="HuggingFace dataset repo ID.")
    parser.add_argument("--no_streaming", action="store_true", default=False,
                        help="Download the full dataset first instead of streaming (default: stream).")
    # Make dataset_base_path optional since we use HF dataset by default
    for action in parser._actions:
        if action.dest == "dataset_base_path":
            action.required = False
            break
    parser.set_defaults(
        dataset_base_path="",
        model_id_with_origin_paths=(
            "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,"
            "Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,"
            "Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth"
        ),
        trainable_models="dit",
        lora_base_model="dit",
        data_file_keys="video",
        gradient_accumulation_steps=1,
        max_pixels=480 * 832,
        num_frames=5,
        output_path="./models/train/Wan2.1-T2V-1.3B_lora",
    )
    args = parser.parse_args()

    streaming = not args.no_streaming
    dataset = create_hf_dataset(
        args.hf_dataset,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        max_pixels=args.max_pixels,
        streaming=streaming,
    )

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )

    if streaming:
        launch_streaming_training(dataset, model, model_logger, args)
    else:
        launch_training_task(dataset, model, model_logger, args=args)
