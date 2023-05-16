import torch
import torch.nn as nn

from diffusers import (AutoencoderKL, DDPMScheduler, UNet2DConditionModel,
                       StableDiffusionPipeline)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAXFormersAttnProcessor,
    LoRAAttnProcessor,
    SlicedAttnAddedKVProcessor,
)
from diffusers.utils.import_utils import is_xformers_available
import logging
from packaging import version
from diffusers.loaders import AttnProcsLayers
import torch.nn.functional as F
from diffusers.utils import TEXT_ENCODER_TARGET_MODULES
from transformers import AutoTokenizer, PretrainedConfig


def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation  # noqa

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


class DreamDiffusionLoRA(nn.Module):
    """Dream Diffusion Model with LoRA.

    Args:
        tokenizer_name (str): Name of tokenizer.
        pretrained_model_name_or_path (str): Path to pretrained model or model
            identifier from huggingface.co/models.
        enable_xformers_memory_efficient_attention (bool, optional): Whether to
            use XFormers for memory efficient attention computation.
        train_text_encoder (bool, optional): Whether to train text encoder.
        with_prior_preservation (bool, optional): Flag to add prior
            preservation loss.
        prior_loss_weight (float, optional): Weight of prior preservation loss.
        revision (str, optional): Revision of pretrained model identifier from
            huggingface.co/models.
    """

    def __init__(self,
                 tokenizer_name: str,
                 pretrained_model_name_or_path: str,
                 enable_xformers_memory_efficient_attention: bool = False,
                 train_text_encoder: bool = False,
                 with_prior_preservation: bool = False,
                 prior_loss_weight: float = 1.0,
                 revision: str = None) -> None:
        super(DreamDiffusionLoRA, self).__init__()
        # Load scheduler, tokenizer and models.

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler")
        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, revision=revision, use_fast=False)
        elif pretrained_model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=revision,
                use_fast=False,
            )
            # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(
            pretrained_model_name_or_path, revision)
        self.text_encoder = text_encoder_cls.from_pretrained(
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
        if enable_xformers_memory_efficient_attention:
            self._set_xformers()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.train_text_encoder = train_text_encoder
        self.with_prior_preservation = with_prior_preservation
        self.prior_loss_weight = prior_loss_weight
        # initialize lora layers
        self._set_unet_lora_layers()
        if train_text_encoder:
            self._set_text_encoder_lora_layers()

    def forward(self,
                pixel_values: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                noise_offset: float = 0) -> torch.Tensor:
        """Forward pass.

        Args:
            pixel_values (torch.Tensor): Raw pixel values of images.
            encoder_hidden_states (torch.Tensor): Text Embeddings.
            noise_offset (bool, optional): The scale of noise offset.
                https://www.crosslabs.org//blog/diffusion-with-offset-noise

        Returns:
            torch.Tensor: Loss.
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

        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps,
                               encoder_hidden_states).sample

        # if model predicts variance, throw away the prediction. we will only
        # train on the simplified training objective. This means that all
        # schedulers using the fine tuned model must be configured to use one
        # of the fixed variance variance types.
        if model_pred.shape[1] == 6:
            model_pred, _ = torch.chunk(model_pred, 2, dim=1)

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

        if self.with_prior_preservation:
            # Chunk the noise and model_pred into two parts and compute the
            # loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = F.mse_loss(
                model_pred.float(), target.float(), reduction="mean")

            # Compute prior loss
            prior_loss = F.mse_loss(
                model_pred_prior.float(),
                target_prior.float(),
                reduction="mean")

            # Add the prior loss to the instance loss.
            loss = loss + self.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(
                model_pred.float(), target.float(), reduction="mean")

        return loss

    def _set_unet_lora_layers(self) -> None:
        """Initialize LoRA layers for UNet2DConditionModel."""
        # Set correct lora layers
        unet_lora_attn_procs = {}
        for name, attn_processor in self.unet.attn_processors.items():
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

            if isinstance(attn_processor,
                          (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor,
                           AttnAddedKVProcessor2_0)):
                lora_attn_processor_class = LoRAXFormersAttnProcessor
            else:
                lora_attn_processor_class = LoRAAttnProcessor

            unet_lora_attn_procs[name] = lora_attn_processor_class(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim)
        self.unet.set_attn_processor(unet_lora_attn_procs)
        self.unet_lora_layers = AttnProcsLayers(self.unet.attn_processors)

    def _set_text_encoder_lora_layers(self) -> None:
        """Initialize LoRA layers for text encoder."""
        # The text encoder comes from 🤗 transformers, so we cannot directly modify it.  # noqa
        # So, instead, we monkey-patch the forward calls of its attention-blocks. For this,  # noqa
        # we first load a dummy pipeline with the text encoder and then do the monkey-patching.  # noqa
        text_encoder_lora_layers = None
        text_lora_attn_procs = {}
        for name, module in self.text_encoder.named_modules():
            if any(x in name for x in TEXT_ENCODER_TARGET_MODULES):
                text_lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=module.out_features, cross_attention_dim=None)
        text_encoder_lora_layers = AttnProcsLayers(text_lora_attn_procs)
        temp_pipeline = StableDiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path, text_encoder=self.text_encoder)
        temp_pipeline._modify_text_encoder(text_lora_attn_procs)
        self.text_encoder = temp_pipeline.text_encoder
        del temp_pipeline

        self.text_encoder_lora_layers = text_encoder_lora_layers

    def _set_xformers(self) -> None:
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

    def _get_tokenizer(self):
        """Return tokenizer"""
        return self.tokenizer

    def _del_tokenizer_text_encoder(self):
        """Delete tokenizer and text encoder."""
        del self.tokenizer
        del self.text_encoder


if __name__ == '__main__':
    model = DreamDiffusionLoRA(
        'runwayml/stable-diffusion-v1-5',
        train_text_encoder=True,
        with_prior_preservation=True,
        prior_loss_weight=1.0)
    pixel_values = torch.randn(8, 3, 224, 224)
    encoder_hidden_states = torch.randn(8, 100, 768)
    loss = model(pixel_values, encoder_hidden_states)
    print(loss.item())
