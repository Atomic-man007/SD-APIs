import gc
import json
import torch
import random
import requests
from PIL import Image
from io import BytesIO
from loguru import logger
from typing import Optional
from datetime import datetime
from dataclasses import dataclass
from PIL.PngImagePlugin import PngInfo
from diffusers import StableDiffusionInpaintPipeline

# utils
import utils

@dataclass
class Inpainting:
    model: Optional[str] = None
    device: Optional[str] = None
    output_path: Optional[str] = None

    def __str__(self) -> str:
        return f"Inpainting(model={self.model}, device={self.device}, output_path={self.output_path})"

    def __post_init__(self):
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_auth_token=utils.use_auth_token(),
        )

        self.pipeline.to(self.device)
        self.pipeline.safety_checker = utils.no_safety_checker
        self._compatible_schedulers = self.pipeline.scheduler.compatibles
        self.scheduler_config = self.pipeline.scheduler.config
        self.compatible_schedulers = {scheduler.__name__: scheduler for scheduler in self._compatible_schedulers}


    def _set_scheduler(self, scheduler_name):
        scheduler = self.compatible_schedulers[scheduler_name].from_config(self.scheduler_config)
        self.pipeline.scheduler = scheduler

    def generate_image(
        self, prompt, negative_prompt, image, mask, guidance_scale, scheduler, steps, seed, height, width, num_images
    ):

        if seed == -1:
            # generate random seed
            seed = random.randint(0, 999999)

        self._set_scheduler(scheduler)
        logger.info(self.pipeline.scheduler)

        if self.device == "mps":
            generator = torch.manual_seed(seed)
            num_images = 1
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        output_images = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
            height=height,
            width=width,
        ).images
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "scheduler": scheduler,
            "steps": steps,
            "seed": seed,
        }
        metadata = json.dumps(metadata)
        _metadata = PngInfo()
        _metadata.add_text("inpainting", metadata)

        utils.save_to_local(
            images=output_images,
            module="inpainting",
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            metadata=metadata,
            output_path=self.output_path,
        )

        torch.cuda.empty_cache()
        gc.collect()
        return output_images, _metadata
