import gc
import json
from dataclasses import dataclass
from typing import Optional

import streamlit as st
import torch
from diffusers import StableDiffusionUpscalePipeline
from loguru import logger
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from datetime import datetime
import utils

@dataclass
class Upscaler:
    model: Optional[str] = None
    device: Optional[str] = None
    output_path: Optional[str] = None

    def __str__(self) -> str:
        return f"Upscaler(model={self.model}, device={self.device}, output_path={self.output_path})"

    def __post_init__(self):
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            self.model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
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
        self, image, prompt, negative_prompt, guidance_scale, noise_level, num_images, eta, scheduler, steps, seed
    ):
        self._set_scheduler(scheduler)
        logger.info(self.pipeline.scheduler)
        if self.device == "mps":
            generator = torch.manual_seed(seed)
            num_images = 1
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        num_images = int(num_images)
        output_images = self.pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            noise_level=noise_level,
            num_inference_steps=steps,
            eta=eta,
            num_images_per_prompt=num_images,
            generator=generator,
            guidance_scale=guidance_scale,
        ).images

        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "noise_level": noise_level,
            "num_images": num_images,
            "eta": eta,
            "scheduler": scheduler,
            "steps": steps,
            "seed": seed,
        }

        metadata = json.dumps(metadata)
        _metadata = PngInfo()
        _metadata.add_text("upscaler", metadata)

        utils.save_to_local(
            images=output_images,
            module="upscaler",
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            metadata=metadata,
            output_path=self.output_path,
        )
        torch.cuda.empty_cache()
        gc.collect()
        return output_images, _metadata