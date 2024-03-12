# importing libraries
import gc
import cv2
import json
import torch
import random
import numpy as np
from PIL import Image
from io import BytesIO
from loguru import logger
from typing import Optional
from datetime import datetime
from dataclasses import dataclass
from PIL.PngImagePlugin import PngInfo
from controlnet_aux import HEDdetector, OpenposeDetector
from diffusers import ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetPipeline

# utils
import utils

# Openpose class object 
@dataclass
class Openpose:
    # Optional parameters for the Openpose class
    model: Optional[str] = None
    device: Optional[str] = None
    output_path: Optional[str] = None

    # String representation of the Openpose object
    def __str__(self) -> str:
        return f"Openpose (model={self.model}, device={self.device}, output_path={self.output_path})"

    # Initialization method for the Openpose class
    def __post_init__(self):
        # Initialize the ControlNet model and the StableDiffusionControlNetPipeline
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16)
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.model, controlnet=self.controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_auth_token=utils.use_auth_token(),
        )

        # Set up device and other configurations
        self.pipeline.to(self.device)
        self.pipeline.safety_checker = utils.no_safety_checker
        self._compatible_schedulers = self.pipeline.scheduler.compatibles
        self.scheduler_config = self.pipeline.scheduler.config
        self.compatible_schedulers = {scheduler.__name__: scheduler for scheduler in self._compatible_schedulers}

    # Method to generate image using the Openpose class
    def generate_image(
        self, 
        prompt, 
        negative_prompt, 
        image, 
        guidance_scale, 
        steps, 
        seed, 
        num_images: int=1,
    ):
        # Generate a random seed if not provided
        if seed == -1:
            seed = random.randint(0, 999999)

        # Set up generator based on the device
        if self.device == "mps":
            generator = torch.manual_seed(seed)
            num_images = 1
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Initialize OpenposeDetector processor
        self.processor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

        # Configure scheduler and enable various pipeline features
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        logger.info(self.pipeline.scheduler)
        self.pipeline.enable_xformers_memory_efficient_attention()
        self.pipeline.enable_model_cpu_offload()

        # Detect pose using the OpenposeDetector
        detected_pose = self.processor(image)

        # Generate images using the pipeline
        output_images = self.pipeline(
            image=detected_pose,
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
        ).images

        # Create metadata for the generated images
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "seed": seed,
        }
        metadata = json.dumps(metadata)
        _metadata = PngInfo()
        _metadata.add_text("openpose", metadata)

        # Save generated images and metadata to the specified output path
        utils.save_to_local_controlnet(
            images=output_images,
            poses=detected_pose,
            module="openpose",
            current_datetime=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            metadata=metadata,
            output_path=self.output_path,
        )

        # Clear CUDA memory and perform garbage collection
        torch.cuda.empty_cache()
        gc.collect()

        # Return generated images and metadata
        return output_images, _metadata
