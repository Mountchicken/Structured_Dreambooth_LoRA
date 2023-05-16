import torch
from torch.utils.data import Dataset
from pathlib import Path
from diffusers import DiffusionPipeline
from tqdm.auto import tqdm
import hashlib


class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on
    multiple GPUs.

    Args:
        prompt (str): The prompt to generate the images.
        num_samples (int): The number of samples to generate.
    """

    def __init__(self, prompt: str, num_samples: int) -> None:
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict:
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def generate_pp_images(accelerator, class_img_root: dir, num_images: int,
                       prompt: str, pretrained_model_name_or_path: str,
                       precision: str, sample_batch_size: int,
                       revision: str) -> None:
    """Generate images for prior preservation.

    Args:
        accelerator (Accelertor): The accelerator to use.
        class_img_root (dir): The root directory to save the images.
        num_images (int): The number of images to generate. If class_img_root
            contains more than num_images, no images will be generated.
        prompt (str): The prompt to generate the images.
        pretrained_model_name_or_path (str): The path to the pretrained model.
        precision (str): The precision to use. Choice in ["fp32", "fp16,
            'bf16"]
        sample_batch_size (int): The batch size to use for sampling.
        revision (str): The revision of the model to use.
    """
    class_images_dir = Path(class_img_root)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_num_images = len(list(class_images_dir.iterdir()))

    if cur_class_num_images < num_images:
        torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32  # noqa
        if precision == "fp32":
            torch_dtype = torch.float32
        elif precision == "fp16":
            torch_dtype = torch.float16
        elif precision == "bf16":
            torch_dtype = torch.bfloat16
        pipeline = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            revision=revision,
        )
        pipeline.set_progress_bar_config(disable=True)
        num_new_images = num_images - cur_class_num_images

        sample_dataset = PromptDataset(prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(
            sample_dataset, batch_size=sample_batch_size)

        sample_dataloader = accelerator.prepare(sample_dataloader)
        pipeline.to(accelerator.device)

        for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process):
            images = pipeline(example["prompt"]).images

            for i, image in enumerate(images):
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = class_images_dir / f"{example['index'][i] + cur_class_num_images}-{hash_image}.jpg"  # noqa
                image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
