import torch
import torch.nn as nn

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.utils.import_utils import is_xformers_available
import logging
from packaging import version
from diffusers.loaders import AttnProcsLayers
import torch.nn.functional as F


class StableDiffusion_LoRA(nn.Module):
    """Stable Diffusion Model with LoRA.

    Args:
        pretrained_model_name_or_path (str): Path to pretrained model or model
            identifier from huggingface.co/models.
        enable_xformers_memory_efficient_attention (bool, optional): Whether to
            use XFormers for memory efficient attention computation.
        revision (str, optional): Revision of pretrained model identifier from
            huggingface.co/models.
    """

    def __init__(self,
                 pretrained_model_name_or_path: str,
                 enable_xformers_memory_efficient_attention: bool = False,
                 revision: str = None):
        super(StableDiffusion_LoRA, self).__init__()
        # Load scheduler, tokenizer and models.

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=revision)
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=revision)
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", revision=revision)
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", revision=revision)
        # freeze parameters of models to save more memory
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self._set_lora_layers()
        if enable_xformers_memory_efficient_attention:
            self._set_xformers()
        self.lora_layers = AttnProcsLayers(self.unet.attn_processors)

    def forward(self,
                pixel_values: torch.Tensor,
                input_ids: torch.Tensor,
                noise_offset: float = 0):
        """Forward pass.

        Args:
            pixel_values (torch.Tensor): Raw pixel values of images.
            input_ids (torch.Tensor): Input ids of text.
            noise_offset (bool, optional): The scale of noise offset.
                https://www.crosslabs.org//blog/diffusion-with-offset-noise
        """
        latents = self.vae.encode(
            pixel_values).latent_dist.sample()  # (N, 4, _, _) # noqa
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1),
                device=latents.device)

        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps, (bsz, ),
            device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep  # noqa
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise,
                                                       timesteps)
        encoder_hidden_states = self.text_encoder(input_ids)[0]

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise,
                                                       timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"  # noqa
            )

        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps,
                               encoder_hidden_states).sample
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss

    def _set_lora_layers(self):
        """Initialize LoRA layers for UNet2DConditionModel."""
        # Set correct lora layers
        lora_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith(
                "attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(
                    reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim)

        self.unet.set_attn_processor(lora_attn_procs)

    def _set_xformers(self):
        """Initialize XFormers for faster attention computation."""
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logging.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs."
                    "If you observe problems during training, please update"
                    "xFormers to at least 0.0.17.")
            self.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"  # noqa
            )


if __name__ == '__main__':
    model = StableDiffusion_LoRA('runwayml/stable-diffusion-v1-5')
    pixel_values = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(0, 10000, (8, 10))
    loss = model(pixel_values, input_ids)
    print(loss.item())
