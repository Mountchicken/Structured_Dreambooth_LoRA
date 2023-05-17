import torch
import numpy as np
from accelerate import Accelerator
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


def validation(model: torch.nn.Module,
               accelerator: Accelerator,
               weight_dtype: torch.dtype,
               epoch: int,
               tb_writer: torch.utils.tensorboard.SummaryWriter,
               args: dict,
               validation_prompt_encoder_hidden_states=None,
               validation_prompt_negative_prompt_embeds=None):
    """Forward inference for validation.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        accelerator (Accelerator): The accelerator to be used.
        weight_dtype (torch.dtype): The dtype of the model weights.
        epoch (int): The current epoch.
        tb_writer (torch.utils.tensorboard.SummaryWriter): The tensorboard
            writer.
        args (dict): The arguments.
        validation_prompt_encoder_hidden_states (torch.Tensor, optional): The
            hidden states of the prompt encoder. Defaults to None.
        validation_prompt_negative_prompt_embeds (torch.Tensor, optional): The
            negative prompt embeddings. Defaults to None.
    """
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=accelerator.unwrap_model(model.unet),
        text_encoder=None if args.pre_compute_text_embeddings else
        accelerator.unwrap_model(model.text_encoder),
        revision=args.revision,
        torch_dtype=weight_dtype,
    )

    # We train on the simplified learning objective. If we were previously
    # predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config, **scheduler_args)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(
        args.seed) if args.seed else None
    if args.pre_compute_text_embeddings:
        pipeline_args = {
            "prompt_embeds": validation_prompt_encoder_hidden_states,
            "negative_prompt_embeds": validation_prompt_negative_prompt_embeds,
        }
    else:
        pipeline_args = {"prompt": args.validation_prompt}
    images = [
        pipeline(**pipeline_args, generator=generator).images[0]
        for _ in range(args.num_validation_images)
    ]
    # save images to file
    for i, img in enumerate(images):
        img.save(f"{args.output_dir}/validation_images/epoch_{epoch}_{i}.png",
                 "PNG")
    # stack images and write to tensorboard
    np_images = np.stack([np.asarray(img) for img in images])
    tb_writer.add_images(
        "validation_images", np_images, epoch, dataformats="NHWC")
    del pipeline
    torch.cuda.empty_cache()
