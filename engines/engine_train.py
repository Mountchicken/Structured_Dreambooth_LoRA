import torch
import os
import logging
import itertools
from tqdm import tqdm
from accelerate import Accelerator
from utils import encode_prompt

from diffusers import DiffusionPipeline
from diffusers.loaders import LoraLoaderMixin, AttnProcsLayers


def save_model_hook(models: torch.nn.Module, weights, output_dir: str):
    """Custom saving hook so that `accelerator.save_state(...)` serializes in a
    nice format.

    Args:
        accelerator (Accelerator): The accelerator to be used.
        model (torch.nn.Module): model to be saved.
        weights (list): The weights to be saved.
        output_dir (str): The output directory.
    """
    # there are only two options here. Either are just the unet attn processor layers
    # or there are the unet and text encoder atten layers
    unet_lora_layers_to_save = None
    text_encoder_lora_layers_to_save = None
    if len(models) > 1:
        with_text_encoder = True
    else:
        with_text_encoder = False

    unet_lora_layers = models[0]
    unet_keys = unet_lora_layers.state_dict().keys()

    text_encoder_lora_layers = models[1] if with_text_encoder else None
    text_encoder_keys = text_encoder_lora_layers.state_dict().keys(
    ) if with_text_encoder else None

    for model in models:
        state_dict = model.state_dict()

        if (text_encoder_lora_layers is not None
                and text_encoder_keys is not None
                and state_dict.keys() == text_encoder_keys):
            # text encoder
            text_encoder_lora_layers_to_save = state_dict
        elif state_dict.keys() == unet_keys:
            # unet
            unet_lora_layers_to_save = state_dict

        # make sure to pop weight so that corresponding model is not saved again
        weights.pop()

    LoraLoaderMixin.save_lora_weights(
        output_dir,
        unet_lora_layers=unet_lora_layers_to_save,
        text_encoder_lora_layers=text_encoder_lora_layers_to_save,
    )


def load_model_hook(models, input_dir, args, weight_dtype):
    # Note we DON'T pass the unet and text encoder here an purpose
    # so that the we don't accidentally override the LoRA layers of
    # unet_lora_layers and text_encoder_lora_layers which are stored in `models`  # noqa
    # with new torch.nn.Modules / weights. We simply use the pipeline class as
    # an easy way to load the lora checkpoints
    temp_pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    temp_pipeline.load_lora_weights(input_dir)

    # load lora weights into models
    models[0].load_state_dict(
        AttnProcsLayers(temp_pipeline.unet.attn_processors).state_dict())
    if len(models) > 1:
        models[1].load_state_dict(
            AttnProcsLayers(
                temp_pipeline.text_encoder_lora_attn_procs).state_dict())

    # delete temporary pipeline and pop models
    del temp_pipeline
    for _ in range(len(models)):
        models.pop()


def train_one_epoch(accelerator: Accelerator, model: torch.nn.Module,
                    train_dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    lr_scheduler: torch.optim.lr_scheduler,
                    tb_writer: torch.utils.tensorboard.SummaryWriter,
                    epoch: int, first_epoch: int, resume_step: int,
                    global_step: int, progress_bar: tqdm, args: dict,
                    weight_dtype: torch.dtype) -> None:
    """Train for one epoch.

    Args:
        accelerator (Accelerator): The accelerator to be used for distributed
            training.
        model (torch.nn.Module): The model to be trained.
        train_dataloader (torch.utils.data.DataLoader): The dataloader for
            training.
        optimizer (torch.optim.Optimizer): The optimizer to be used for
            training.
        lr_scheduler (torch.optim.lr_scheduler): The learning rate scheduler to
            be used for training.
        tb_writer (torch.utils.tensorboard.SummaryWriter): The tensorboard
            writer to be used for logging.
        epoch (int): The current epoch.
        first_epoch (int): The first epoch.
        resume_step (int): The step to resume from.
        global_step (int): The current global step.
        progress_bar (tqdm): The progress bar to be used for logging.
        args (dict): The arguments to be used for training.
        weight_dtype (torch.dtype): The data type to be used for weights.
    """
    model.unet.train()
    if args.train_text_encoder:
        model.text_encoder.train()
    for step, batch in enumerate(train_dataloader):
        # Skip steps until we reach the resumed step
        if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:  # noqa
            if step % args.gradient_accumulation_steps == 0:
                progress_bar.update(1)
            continue

        with accelerator.accumulate(model.unet):
            pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
            if args.pre_compute_text_embeddings:
                encoder_hidden_states = batch["input_ids"]
            else:
                encoder_hidden_states = encode_prompt(
                    model.text_encoder,
                    batch["input_ids"],
                    batch["attention_mask"],
                    text_encoder_use_attention_mask=args.
                    text_encoder_use_attention_mask,
                )
            loss = model(
                pixel_values, encoder_hidden_states=encoder_hidden_states)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = (
                    itertools.chain(
                        model.unet_lora_layers.parameters(),
                        model.text_encoder_lora_layers.parameters())
                    if args.train_text_encoder else
                    model.unet_lora_layers.parameters())
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind
        # the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                # Save model checkpoint
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir,
                                             f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logging.info(f"Saved state to {save_path}")
            memory_mb = int(torch.cuda.max_memory_allocated() /
                            (1024.0 * 1024.0))
            log_info = f'Epoch: {epoch}/{args.num_train_epochs}' + \
                       f' | Step: {global_step}/{args.max_train_steps}' + \
                       f' | Loss: {loss.detach().item():.5f}' + \
                       f' | LR: {lr_scheduler.get_last_lr()[0]:.8f}' + \
                       f' | Memory: {memory_mb}MiB'
            logging.info(log_info)
        logs = {
            "loss": f'{loss.detach().item():.5f}',
            "lr": f'{lr_scheduler.get_last_lr()[0]:.8f}',
            "memory": memory_mb,
        }
        progress_bar.set_postfix(**logs)
        tb_writer.add_scalar(
            "lr", lr_scheduler.get_last_lr()[0], global_step=global_step)
        tb_writer.add_scalar(
            "loss", loss.detach().item(), global_step=global_step)
        if global_step >= args.max_train_steps:
            break
    return global_step
