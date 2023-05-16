
def save_model_hook(accelerator, models, weights, output_dir):
    # there are only two options here. Either are just the unet attn processor layers
    # or there are the unet and text encoder atten layers
    unet_lora_layers_to_save = None
    text_encoder_lora_layers_to_save = None

    if args.train_text_encoder:
        text_encoder_keys = accelerator.unwrap_model(
            text_encoder_lora_layers).state_dict().keys()
    unet_keys = accelerator.unwrap_model(
        unet_lora_layers).state_dict().keys()

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

def load_model_hook(models, input_dir):
    # Note we DON'T pass the unet and text encoder here an purpose
    # so that the we don't accidentally override the LoRA layers of
    # unet_lora_layers and text_encoder_lora_layers which are stored in `models`
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