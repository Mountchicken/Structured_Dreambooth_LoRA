import argparse
import os
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of an inference script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='runwayml/stable-diffusion-v1-5',
        help=  # noqa
        "Path to pretrained model or model identifier from huggingface.co/models.",  # noqa
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=
        'work_dirs/test_peroson/no_prior_no_train_text_lr_2e-4/20230518_143106/checkpoint-1050',
        help="Path to checkpoint directory.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default='no',
        help=  # noqa
        (
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="  # noqa
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"  # noqa
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."  # noqa
        ))
    parser.add_argument(
        "--prompt",
        type=str,
        default='a photo of a sks person taking selfie at eiffel tower',  # noqa
        help="prompt used for generation",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="number of inference steps")
    parser.add_argument(
        "--num_images",
        type=int,
        default=16,
        help="number of images to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=
        "work_dirs/test_person/no_prior_no_train_text_lr_2e-4/inference_600EP_apple",
        help="The output directory where the generated image will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # pring args
    print(args)
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype)

    # We train on the simplified learning objective. If we were
    # previously predicting a variance, we need the scheduler
    # to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipeline.scheduler = DDIMScheduler.from_config(
        pipeline.scheduler.config, **scheduler_args)

    pipeline = pipeline.to(args.device)

    # load attention processors
    pipeline.load_lora_weights(args.checkpoint_dir)

    # run inference
    images = []
    generator = torch.Generator(
        device=args.device).manual_seed(args.seed) if args.seed else None
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for i in range(args.num_images):
        image = pipeline(
            args.prompt,
            num_inference_steps=args.num_inference_steps,
            generator=generator).images[0]
        image.save(f"{args.output_dir}/{i}.png", "PNG")
