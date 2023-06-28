import base64
import gc
import json
import os
import random
import tempfile
from dataclasses import dataclass
from io import BytesIO
from typing import Optional
from datetime import datetime
import requests
import streamlit as st
import torch
from diffusers import (
    AltDiffusionImg2ImgPipeline,
    AltDiffusionPipeline,
    DiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
)
from loguru import logger
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import utils
from transformers import CLIPTokenizer, CLIPTextModel




def load_embed(learned_embeds_path, text_encoder, tokenizer, token=None):
    # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
    if len(loaded_learned_embeds) > 2:
        embeds = loaded_learned_embeds["string_to_param"]["*"][-1, :]
    else:
        # separate token and the embeds
        trained_token = list(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token]
    token = None
    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    i = 1
    while num_added_tokens == 0:
        print(f"The tokenizer already contains the token {token}.")
        token = f"{token[:-1]}-{i}>"
        print(f"Attempting to add the token {token}.")
        num_added_tokens = tokenizer.add_tokens(token)
        i += 1

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return token

@dataclass
class X2Image:
    device: Optional[str] = None
    model: Optional[str] = None
    output_path: Optional[str] = None
    custom_pipeline: Optional[str] = None
    embeddings_url: Optional[str] = None
    token_identifier: Optional[str] = None

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    def __str__(self) -> str:
        return f"X2Image(model={self.model}, pipeline={self.custom_pipeline})"

    def __post_init__(self):
        if self.model.endswith(".safetensors"):
            self.text2img_pipeline = StableDiffusionPipeline.from_ckpt(
                self.model,
                local_files_only=True,
                safety_checker=None,
                requires_safety_checker=False,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                custom_pipeline=self.custom_pipeline,
                # use_auth_token=utils.use_auth_token(),
            )
        else:
            self.text2img_pipeline = DiffusionPipeline.from_pretrained(
                self.model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                custom_pipeline=self.custom_pipeline,
                # use_auth_token=utils.use_auth_token(),
            )

        model_id = "timbrooks/instruct-pix2pix"
        components = self.text2img_pipeline.components
        self.pix2pix_pipeline = None
        if isinstance(self.text2img_pipeline, StableDiffusionPipeline):
            self.img2img_pipeline = StableDiffusionImg2ImgPipeline(**components)
            self.pix2pix_pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")
        elif isinstance(self.text2img_pipeline, AltDiffusionPipeline):
            self.img2img_pipeline = AltDiffusionImg2ImgPipeline(**components)
        else:
            self.img2img_pipeline = None
            logger.error("Model type not supported, img2img pipeline not created")

        self.text2img_pipeline.to(self.device)
        self.text2img_pipeline.safety_checker = None
        self.text2img_pipeline.requires_safety_checker = False
        self.img2img_pipeline.to(self.device)
        self.img2img_pipeline.safety_checker = utils.no_safety_checker
        if self.pix2pix_pipeline is not None:
            self.pix2pix_pipeline.to(self.device)
            self.pix2pix_pipeline.safety_checker = utils.no_safety_checker

        self.compatible_schedulers = {
            scheduler.__name__: scheduler for scheduler in self.text2img_pipeline.scheduler.compatibles
        }

        if len(self.embeddings_url) > 0 and len(self.token_identifier) > 0:
            # download the embeddings
            self.embeddings_path = self.embeddings_url
            load_embed(
                learned_embeds_path=self.embeddings_path,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                token=self.token_identifier,
            )
        if len(self.embeddings_url) > 0 and len(self.token_identifier) > 0:
            # download the embeddings
            self.embeddings_path = self.embeddings_url
            load_embed(
                learned_embeds_path=self.embeddings_path,
                text_encoder=self.img2img_pipeline.text_encoder,
                tokenizer=self.img2img_pipeline.tokenizer,
                token=self.token_identifier,
            )

    def _set_scheduler(self, pipeline_name, scheduler_name):
        if pipeline_name == "text2img":
            scheduler_config = self.text2img_pipeline.scheduler.config
        elif pipeline_name == "img2img":
            scheduler_config = self.img2img_pipeline.scheduler.config
        elif pipeline_name == "pix2pix":
            scheduler_config = self.pix2pix_pipeline.scheduler.config
        else:
            raise ValueError(f"Pipeline {pipeline_name} not supported")

        scheduler = self.compatible_schedulers[scheduler_name].from_config(scheduler_config)

        if pipeline_name == "text2img":
            self.text2img_pipeline.scheduler = scheduler
        elif pipeline_name == "img2img":
            self.img2img_pipeline.scheduler = scheduler
        elif pipeline_name == "img2img":
            self.pix2pix_pipeline.scheduler = scheduler

    def _pregen(self, pipeline_name, scheduler, num_images, seed):
        self._set_scheduler(scheduler_name=scheduler, pipeline_name=pipeline_name)
        if self.device == "mps":
            generator = torch.manual_seed(seed)
            num_images = 1
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        num_images = int(num_images)
        return generator, num_images

    def _postgen(self, metadata, output_images, pipeline_name):
        torch.cuda.empty_cache()
        gc.collect()
        metadata = json.dumps(metadata)
        _metadata = PngInfo()
        _metadata.add_text(pipeline_name, metadata)
        utils.save_to_local(
            images=output_images,
            module=pipeline_name,
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            metadata=metadata,
            output_path=self.output_path,
        )
        return output_images, _metadata

    def text2img_generate(
        self, prompt, negative_prompt, scheduler, image_size, num_images, guidance_scale, steps, seed
    ):

        if seed == -1:
            # generate random seed
            seed = random.randint(0, 999999)

        generator, num_images = self._pregen(
            pipeline_name="text2img",
            scheduler=scheduler,
            num_images=num_images,
            seed=seed,
        )
        output_images = self.text2img_pipeline(
            prompt,
            negative_prompt=negative_prompt,
            width=image_size[1],
            height=image_size[0],
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        ).images
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "scheduler": scheduler,
            "image_size": image_size,
            "num_images": num_images,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "seed": seed,
        }

        output_images, _metadata = self._postgen(
            metadata=metadata,
            output_images=output_images,
            pipeline_name="text2img",
        )
        return output_images, _metadata

    def img2img_generate(
        self, prompt, image, strength, negative_prompt, scheduler, num_images, guidance_scale, steps, seed
    ):

        if seed == -1:
            # generate random seed
            seed = random.randint(0, 999999)

        generator, num_images = self._pregen(
            pipeline_name="img2img",
            scheduler=scheduler,
            num_images=num_images,
            seed=seed,
        )
        output_images = self.img2img_pipeline(
            prompt=prompt,
            image=image,
            strength=strength,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        ).images
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "scheduler": scheduler,
            "num_images": num_images,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "seed": seed,
        }
        output_images, _metadata = self._postgen(
            metadata=metadata,
            output_images=output_images,
            pipeline_name="img2img",
        )
        return output_images, _metadata

    def pix2pix_generate(
        self, prompt, image, negative_prompt, scheduler, num_images, guidance_scale, image_guidance_scale, steps, seed
    ):
        if seed == -1:
            # generate random seed
            seed = random.randint(0, 999999)

        generator, num_images = self._pregen(
            pipeline_name="pix2pix",
            scheduler=scheduler,
            num_images=num_images,
            seed=seed,
        )
        output_images = self.pix2pix_pipeline(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        ).images
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "scheduler": scheduler,
            "num_images": num_images,
            "guidance_scale": guidance_scale,
            "image_guidance_scale": image_guidance_scale,
            "steps": steps,
            "seed": seed,
        }
        output_images, _metadata = self._postgen(
            metadata=metadata,
            output_images=output_images,
            pipeline_name="pix2pix",
        )
        return output_images, _metadata