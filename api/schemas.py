from typing import Dict, List

from pydantic import BaseModel, Field


class Text2ImgParams(BaseModel):
    prompt: str = Field(..., description="Text prompt for the model")
    negative_prompt: str = Field(None, description="Negative text prompt for the model")
    scheduler: str = Field("EulerAncestralDiscreteScheduler", description="Scheduler to use for the model")
    image_height: int = Field(512, description="Image height")
    image_width: int = Field(512, description="Image width")
    num_images: int = Field(1, description="Number of images to generate")
    guidance_scale: float = Field(7, description="Guidance scale")
    steps: int = Field(50, description="Number of steps to run the model for")
    seed: int = Field(42, description="Seed for the model")


class Img2ImgParams(BaseModel):
    prompt: str = Field(..., description="Text prompt for the model")
    negative_prompt: str = Field(None, description="Negative text prompt for the model")
    scheduler: str = Field("EulerAncestralDiscreteScheduler", description="Scheduler to use for the model")
    strength: float = Field(0.7, description="Strength")
    num_images: int = Field(1, description="Number of images to generate")
    guidance_scale: float = Field(7, description="Guidance scale")
    steps: int = Field(50, description="Number of steps to run the model for")
    seed: int = Field(42, description="Seed for the model")


class InstructPix2PixParams(BaseModel):
    prompt: str = Field(..., description="Text prompt for the model")
    negative_prompt: str = Field(None, description="Negative text prompt for the model")
    scheduler: str = Field("EulerAncestralDiscreteScheduler", description="Scheduler to use for the model")
    num_images: int = Field(1, description="Number of images to generate")
    guidance_scale: float = Field(7, description="Guidance scale")
    image_guidance_scale: float = Field(1.5, description="Image guidance scale")
    steps: int = Field(50, description="Number of steps to run the model for")
    seed: int = Field(42, description="Seed for the model")


class ImgResponse(BaseModel):
    images: List[str] = Field(..., description="List of images in base64 format")
    metadata: Dict = Field(..., description="Metadata")


class InpaintingParams(BaseModel):
    prompt: str = Field(..., description="Text prompt for the model")
    negative_prompt: str = Field(None, description="Negative text prompt for the model")
    scheduler: str = Field("EulerAncestralDiscreteScheduler", description="Scheduler to use for the model")
    image_height: int = Field(512, description="Image height")
    image_width: int = Field(512, description="Image width")
    num_images: int = Field(1, description="Number of images to generate")
    guidance_scale: float = Field(7, description="Guidance scale")
    steps: int = Field(50, description="Number of steps to run the model for")
    seed: int = Field(42, description="Seed for the model")


class TextualInversionParams(BaseModel):
    prompt: str = Field(..., description="Text prompt for the pipeline")
    scheduler: str = Field("DPMSolverMultistepScheduler", description="Scheduler to use for the model")
    negative_prompt: str = Field(None, description="Negative text prompt for the pipeline")
    image_height: int = Field(512, description="Image height")
    image_width: int = Field(512, description="Image width")
    steps: int = Field(50, description="Number of inference steps")
    guidance_scale: float = Field(7, description="Guidance scale")
    num_images: int = Field(1, description="Number of images per prompt")
    seed: int = Field(42, description="Seed for the model")


class CannyParams(BaseModel):
    prompt: str = Field(..., description="Text prompt for the pipeline")
    negative_prompt: str = Field(None, description="Negative text prompt for the pipeline")
    steps: int = Field(50, description="Number of inference steps")
    low_threshold: int = Field(100, description="low_threshold scale")
    high_threshold: int = Field(200, description="high_threshold scale")
    guidance_scale: float = Field(7, description="Guidance scale")
    num_images: int = Field(1, description="Number of images per prompt")
    seed: int = Field(42, description="Seed for the model")

class OpenposeParams(BaseModel):
    prompt: str = Field(..., description="Text prompt for the pipeline")
    negative_prompt: str = Field(None, description="Negative text prompt for the pipeline")
    steps: int = Field(50, description="Number of inference steps")
    guidance_scale: float = Field(7, description="Guidance scale")
    num_images: int = Field(1, description="Number of images per prompt")
    seed: int = Field(42, description="Seed for the model")


class UpscalerParams(BaseModel):
    prompt: str = Field(..., description="Text prompt for the pipeline")
    negative_prompt: str = Field(None, description="Negative text prompt for the pipeline")
    steps: int = Field(50, description="Number of inference steps")
    num_images: int = Field(1, description="Number of images per promp")
    guidance_scale: float = Field(7, description="Guidance scale")
    noise_level: int = Field(7, description="noise level scale")
    eta: float = Field(0.1, description="eta scale")
    seed: int = Field(42, description="Seed for the model")
    scheduler: str = Field("EulerAncestralDiscreteScheduler", description="Scheduler to use for the model")

