import gc
import json
import utils
import torch
import streamlit as st
from loguru import logger
from typing import Optional
from datetime import datetime
from dataclasses import dataclass
from PIL.PngImagePlugin import PngInfo
from diffusers import DiffusionPipeline
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
class TextualInversion:
    model: str
    embeddings_url: str
    token_identifier: str
    device: Optional[str] = None
    output_path: Optional[str] = None
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    def __str__(self) -> str:
        return f"TextualInversion(model={self.model}, embeddings={self.embeddings_url}, token_identifier={self.token_identifier}, device={self.device}, output_path={self.output_path})"

    def __post_init__(self):
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.pipeline.to(self.device)
        self.pipeline.safety_checker = utils.no_safety_checker
        self._compatible_schedulers = self.pipeline.scheduler.compatibles
        self.scheduler_config = self.pipeline.scheduler.config
        self.compatible_schedulers = {scheduler.__name__: scheduler for scheduler in self._compatible_schedulers}

        # download the embeddings
        if len(self.embeddings_url) > 0 and len(self.token_identifier) > 0:
            self.embeddings_path = self.embeddings_url
            load_embed(
                learned_embeds_path=self.embeddings_path,
                text_encoder=self.pipeline.text_encoder,
                tokenizer=self.pipeline.tokenizer,
                token=self.token_identifier,
            )

    def _set_scheduler(self, scheduler_name):
        scheduler = self.compatible_schedulers[scheduler_name].from_config(self.scheduler_config)
        self.pipeline.scheduler = scheduler

    def generate_image(self, prompt, negative_prompt, scheduler, image_size, num_images, guidance_scale, steps, seed):
        self._set_scheduler(scheduler)
        logger.info(self.pipeline.scheduler)
        if self.device == "mps":
            generator = torch.manual_seed(seed)
            num_images = 1
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        num_images = int(num_images)
        output_images = self.pipeline(
            prompt,
            negative_prompt=negative_prompt,
            width=image_size[1],
            height=image_size[0],
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        ).images
        torch.cuda.empty_cache()
        gc.collect()
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
        metadata = json.dumps(metadata)
        _metadata = PngInfo()
        _metadata.add_text("textual_inversion", metadata)

        utils.save_to_local(
            images=output_images,
            module="textual_inversion",
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            metadata=metadata,
            output_path=self.output_path,
        )

        return output_images, _metadata
