import gc
import json
import random
from dataclasses import dataclass
from io import BytesIO
from typing import Optional
from datetime import datetime
import numpy as np
import cv2
import torch
from diffusers import ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetPipeline
from loguru import logger
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# utils
import utils


@dataclass
class Canny:
    model: Optional[str] = None
    device: Optional[str] = None
    output_path: Optional[str] = None

    def __str__(self) -> str:
        return f"Canny (model={self.model}, device={self.device}, output_path={self.output_path})"

    def __post_init__(self):
        self.controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)

        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.model, controlnet=self.controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_auth_token=utils.use_auth_token(),
        )

        self.pipeline.to(self.device)
        self.pipeline.safety_checker = utils.no_safety_checker
        self._compatible_schedulers = self.pipeline.scheduler.compatibles
        self.scheduler_config = self.pipeline.scheduler.config
        self.compatible_schedulers = {scheduler.__name__: scheduler for scheduler in self._compatible_schedulers}


    # def _set_scheduler(self, scheduler_name):
    #     self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
    def generate_image(
        self, 
        prompt, 
        negative_prompt, 
        image, 
        guidance_scale, 
        steps, 
        seed, 
        low_threshold: int=100,
        high_threshold: int=200, 
        num_images: int=1,
    ):

        if seed == -1:
            # generate random seed
            seed = random.randint(0, 999999)
        if self.device == "mps":
            generator = torch.manual_seed(seed)
            num_images = 1
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        logger.info(self.pipeline.scheduler)
        self.pipeline.enable_xformers_memory_efficient_attention()
        # self.pipeline.enable_attention_slicing(1)
        self.pipeline.enable_model_cpu_offload()
        # get canny image
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        output_images = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=canny_image,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
        ).images
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "seed": seed,
        }
        metadata = json.dumps(metadata)
        _metadata = PngInfo()
        _metadata.add_text("canny", metadata)

        utils.save_to_local(
            images=output_images,
            module="canny",
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            metadata=metadata,
            output_path=self.output_path,
        )

        torch.cuda.empty_cache()
        gc.collect()
        return output_images, _metadata