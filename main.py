import argparse
import gc

import logging
import time
import math
import os
import warnings

import torch

import torch.utils.checkpoint
from torch.utils.tensorboard import SummaryWriter
import transformers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from tqdm.auto import tqdm
import diffusers
from diffusers.optimization import get_scheduler

from tools import generate_pp_images
from models import DreamDiffusionLoRA
from utils import compute_text_embeddings, Logger
from datasets import DreamBoothDataset, DB_collate_fn
from engines import (train_one_epoch, validation, save_model_hook,
                     load_model_hook)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='runwayml/stable-diffusion-v1-5',
        # required=True,
        help=  # noqa
        "Path to pretrained model or model identifier from huggingface.co/models.",  # noqa
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=  # noqa
        "Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default='class_images',
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default='A photo of a sks dog',
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default='A photo of a dog',
        help=  # noqa
        "The prompt to specify images in the same class as provided instance images.",  # noqa
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default='A photo of a sks person swimming',
        help=  # noqa
        "A prompt that is used during validation to verify that the model is learning.",  # noqa
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help=  # noqa
        "Number of images that should be generated during validation with `validation_prompt`.",  # noqa
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=4,
        help=  # noqa
        (
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"  # noqa
            " `args.validation_prompt` multiple times: `args.num_validation_images`."  # noqa
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        # default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=40,
        help=  # noqa
        (
            "Minimal class images for prior preservation loss. If there are not enough images already present in"  # noqa
            " class_data_dir, additional images will be sampled with class_prompt."  # noqa
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="work_dirs",
        help=  # noqa
        "The output directory where the model predictions and checkpoints will be written.",  # noqa
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=  # noqa
        (
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"  # noqa
            " resolution"),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help=  # noqa
        (
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"  # noqa
            " cropped. The images will be resized to the resolution first before cropping."  # noqa
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help=  # noqa
        "Whether to train the text encoder. If set, the text encoder should be float32 precision.",  # noqa
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.")
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for sampling images.")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=400,
        help=  # noqa
        "Total number of training steps to perform.  If provided, overrides num_train_epochs.",  # noqa
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=50,
        help=  # noqa
        (
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"  # noqa
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"  # noqa
            " training using `--resume_from_checkpoint`."),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=  # noqa
        (
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."  # noqa
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"  # noqa
            " for more docs"),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=  # noqa
        (
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"  # noqa
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'  # noqa
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=  # noqa
        "Number of updates steps to accumulate before performing a backward/update pass.",  # noqa
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=  # noqa
        "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",  # noqa
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help=  # noqa
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help=  # noqa
        "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",  # noqa
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=  # noqa
        (
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'  # noqa
            ' "constant", "constant_with_warmup"]'),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help=  # noqa
        "Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=  # noqa
        (
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."  # noqa
        ),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use.")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help=  # noqa
        "The name of the repository to keep in sync with the local `output_dir`.",  # noqa
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"  # noqa
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=  # noqa
        (
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"  # noqa
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"  # noqa
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=  # noqa
        (
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'  # noqa
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'  # noqa
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='fp16',
        choices=["no", "fp16", "bf16"],
        help=  # noqa
        (
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="  # noqa
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"  # noqa
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."  # noqa
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default='fp16',
        choices=["no", "fp32", "fp16", "bf16"],
        help=  # noqa
        (
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="  # noqa
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."  # noqa
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.")
    parser.add_argument(
        "--pre_compute_text_embeddings",
        action="store_true",
        help=  # noqa
        "Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`.",  # noqa
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        required=False,
        help=  # noqa
        "The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",  # noqa
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        required=False,
        help="Whether to use attention mask for the text encoder",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError(
                "You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn(
                "You need not use --class_data_dir without --with_prior_preservation."  # noqa
            )
        if args.class_prompt is not None:
            warnings.warn(
                "You need not use --class_prompt without --with_prior_preservation."  # noqa
            )

    if args.train_text_encoder and args.pre_compute_text_embeddings:
        raise ValueError(
            "`--train_text_encoder` cannot be used with `--pre_compute_text_embeddings`"  # noqa
        )

    return args


def main(args):
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.output_dir = os.path.join(args.output_dir, timestamp)

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='all',
        project_dir=args.output_dir,
        project_config=accelerator_project_config,
    )

    Logger.init(args.output_dir, 'log')
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    logging.info("Generating class images for prior preservation.")
    if args.with_prior_preservation:
        generate_pp_images(
            accelerator=accelerator,
            class_img_root=args.class_data_dir,
            num_images=args.num_class_images,
            prompt=args.class_prompt,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            precision=args.prior_generation_precision,
            sample_batch_size=args.sample_batch_size,
            revision=args.revision,
        )

    # Creat working directory
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(
                os.path.join(args.output_dir, 'validation_images'),
                exist_ok=True)
    tb_writer = SummaryWriter(
        log_dir=os.path.join(args.output_dir, 'tensorboard'))
    # Define the model

    model = DreamDiffusionLoRA(
        tokenizer_name=args.tokenizer_name,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        enable_xformers_memory_efficient_attention=args.
        enable_xformers_memory_efficient_attention,
        train_text_encoder=args.train_text_encoder,
        with_prior_preservation=args.with_prior_preservation,
        prior_loss_weight=args.prior_loss_weight,
        revision=args.revision)

    # Change the model's dtype to the one specified by the user.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model = model.to(accelerator.device, dtype=weight_dtype)
    # For LoRA Layers, we need to convert them to float32
    model.unet_lora_layers.to(accelerator.device, dtype=torch.float32)
    if args.train_text_encoder:
        model.text_encoder_lora_layers.to(
            accelerator.device, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices  # noqa
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.train_batch_size * accelerator.num_processes)

    # Define Optimizer, only for parameters that are not frozen
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay)

    # Precompute prompt embeddings if requested. In this case, the text encoder
    # will not be kept in memory during training, which is aslo incompatible
    # with training the text encoder.

    if args.pre_compute_text_embeddings:
        pre_computed_instance_prompt_embeddings = compute_text_embeddings(
            args.instance_prompt)
        validation_prompt_negative_prompt_embeds = compute_text_embeddings("")
        if args.validation_prompt is not None:
            validation_prompt_encoder_hidden_states = compute_text_embeddings(
                args.validation_prompt)
        else:
            validation_prompt_encoder_hidden_states = None
        if args.class_prompt is not None:
            pre_computed_class_prompt_embeddings = compute_text_embeddings(
                args.class_prompt)
        else:
            pre_computed_class_prompt_embeddings = None
        model._del_tokenizer_text_encoder()
        gc.collect()
        torch.cuda.empty_cache()
    else:
        pre_computed_instance_prompt_embeddings = None
        validation_prompt_negative_prompt_embeds = None
        validation_prompt_encoder_hidden_states = None
        pre_computed_class_prompt_embeddings = None

    # Build Dataset and Dataloaer
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir
        if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        class_num=args.num_class_images,
        tokenizer=model._get_tokenizer(),
        img_size=args.resolution,
        center_crop=args.center_crop,
        instance_prompt_encoder_hidden_states=  # noqa
        pre_computed_instance_prompt_embeddings,
        class_prompt_encoder_hidden_states=  # noqa
        pre_computed_class_prompt_embeddings,  # noqa
        tokenizer_max_length=args.tokenizer_max_length,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=DB_collate_fn(args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch  # noqa
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps *
        args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps *
        args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        _, _, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model.unet_lora_layers, model.text_encoder_lora_layers, optimizer,
            train_dataloader, lr_scheduler)
    else:
        _, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model.unet_lora_layers, optimizer, train_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the
    # training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch  # noqa
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps /
                                      num_update_steps_per_epoch)
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps  # noqa
    logging.info("{}".format(args).replace(', ', ',\n'))
    logging.info(
        f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}'  # noqa
    )
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataset)}")
    logging.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logging.info(f"  Num Epochs = {args.num_train_epochs}")
    logging.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logging.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"  # noqa
    )
    logging.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logging.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."  # noqa
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)
    else:
        resume_step = 0
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    for epoch in range(first_epoch, args.num_train_epochs):
        global_step = train_one_epoch(
            accelerator=accelerator,
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            tb_writer=tb_writer,
            epoch=epoch,
            first_epoch=first_epoch,
            resume_step=resume_step,
            global_step=global_step,
            progress_bar=progress_bar,
            args=args,
            weight_dtype=weight_dtype,
        )
        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:  # noqa
                logging.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"  # noqa
                    f" {args.validation_prompt}.")
                validation(
                    model=model,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    epoch=epoch,
                    global_step=global_step,
                    tb_writer=tb_writer,
                    args=args,
                    validation_prompt_encoder_hidden_states=  # noqa
                    validation_prompt_encoder_hidden_states,
                    validation_prompt_negative_prompt_embeds=  # noqa
                    validation_prompt_negative_prompt_embeds)


if __name__ == "__main__":
    args = parse_args()
    main(args)
