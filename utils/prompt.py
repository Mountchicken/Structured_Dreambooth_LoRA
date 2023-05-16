import torch


def tokenize_prompt(tokenizer,
                    prompt: str,
                    tokenizer_max_length: int = None) -> dict:
    """Tokenize the prompt text with the tokenizer.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be used
            for tokenizing the prompts.
        prompt (str): The prompt text.
        tokenizer_max_length (int, optional): The maximum length of the
            tokenizer. Defaults to None.

    Returns:
        dict: A dictionary containing the tokenized prompt text.
    """
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(
        text_encoder,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        text_encoder_use_attention_mask: bool = False) -> torch.Tensor:
    """Given the tokenized prompt text, encode the prompt text with the text
    encoder.

    Args:
        text_encoder (transformers.PreTrainedModel): The text encoder to be
            used for encoding the prompt text.
        input_ids (torch.Tensor): The tokenized prompt text.
        attention_mask (torch.Tensor, optional): The attention mask for the
            tokenized prompt text. Defaults to None.
        text_encoder_use_attention_mask (bool, optional): Whether to use
            attention mask for the text encoder. Defaults to False.

    Returns:
        torch.Tensor: The embedding of the prompt text.
    """
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def compute_text_embeddings(
    tokenizer,
    text_encoder,
    prompt: str,
    tokenizer_max_length: int,
    text_encoder_use_attention_mask: bool = False,
) -> torch.Tensor:
    """Given the prompt text, tokenize and encode the prompt text with the
    tokenizer and text encoder.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be used
            for tokenizing the prompts.
        text_encoder (transformers.PreTrainedModel): The text encoder to be
            used for encoding the prompt text.
        prompt (str): The prompt text.
        tokenizer_max_length (int): The maximum length of the tokenizer.
        text_encoder_use_attention_mask (bool, optional): Whether to use
            attention mask for the text encoder. Defaults to False.

    Returns:
        torch.Tensor: The embedding of the prompt text.
    """
    with torch.no_grad():
        text_inputs = tokenize_prompt(
            tokenizer, prompt, tokenizer_max_length=tokenizer_max_length)
        prompt_embeds = encode_prompt(
            text_encoder,
            text_inputs.input_ids,
            text_inputs.attention_mask,
            text_encoder_use_attention_mask=text_encoder_use_attention_mask)

    return prompt_embeds
