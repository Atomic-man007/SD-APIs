#Libraries
import io
import os
import numpy as np
from PIL import Image
from loguru import logger
from starlette.middleware.cors import CORSMiddleware
from fastapi import Depends, FastAPI, File, UploadFile


#imports
from schemas import Img2ImgParams, OpenposeParams, UpscalerParams, CannyParams, ImgResponse, TextualInversionParams, InpaintingParams, InstructPix2PixParams, Text2ImgParams
from canny import Canny
from x2image import X2Image
from openpose import Openpose
from inpainting import Inpainting
from upcale_image import Upscaler
from api_utils import convert_to_b64_list
from textual_inversion import TextualInversion



app = FastAPI(
    title="Stable diffusion API's"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup_event():

    x2img_model = "runwayml/stable-diffusion-v1-5" #r"C:\Users\srika\SD-APIs\api\chilloutmix_NiPrunedFp32Fix.safetensors"  #"stabilityai/stable-diffusion-2-1"
    x2img_pipeline = os.environ.get("X2IMG_PIPELINE") #custom pipeline
    inpainting_model = "stabilityai/stable-diffusion-2-inpainting"
    textualinversion_model = "runwayml/stable-diffusion-v1-5"
    pix_model = "timbrooks/instruct-pix2pix"
    upscale_model = "stabilityai/stable-diffusion-x4-upscaler"
    canny_model = "runwayml/stable-diffusion-v1-5"
    openpose_model = "SG161222/Realistic_Vision_V2.0"
    device = "cuda"
    output_path = r"C:\Users\srika\SD-APIs\data"
    ti_identifier = "<hitokomoru-style>"
    ti_embeddings_url = r"C:\Users\srika\SD-APIs\api\learned_embeds.bin"
    logger.info("@@@@@ Starting API @@@@@ ")
    logger.info(f"Text2Image Model: {x2img_model}")
    logger.info(f"Text2Image Pipeline: {x2img_pipeline if x2img_pipeline is not None else 'Vanilla'}")
    logger.info(f"Inpainting Model: {inpainting_model}")
    logger.info(f"textual inversion Model: {textualinversion_model}")
    logger.info(f"pix2pix Model: {pix_model}")
    logger.info(f"upscale Model: {upscale_model}")
    logger.info(f"Canny Model: {canny_model}")
    logger.info(f"Openpose Model: {openpose_model}")
    logger.info(f"Device: {device}")
    logger.info(f"Output Path: {output_path}")
    logger.info(f"Token Identifier: {ti_identifier}")
    logger.info(f"Token Embeddings URL: {ti_embeddings_url}")

    logger.info("Loading x2img model...")
    if x2img_model is not None:
        app.state.x2img_model = X2Image(
            model=x2img_model,
            device=device,
            output_path=output_path,
            custom_pipeline=x2img_pipeline,
            token_identifier=ti_identifier,
            embeddings_url=ti_embeddings_url,
        )
    else:
        app.state.x2img_model = None

    logger.info("*=*=*=*=*=*=*=*=*=*=*=*=*")
    logger.info("Loading upscaler model...")
    if upscale_model is not None:
        app.state.upscale_model = Upscaler(
            model=upscale_model,
            device=device,
            output_path=output_path,
        )
    else:
        app.state.upscale_model = None
    logger.info("Loading inpainting model...")
    if inpainting_model is not None:
        app.state.inpainting_model = Inpainting(
            model=inpainting_model,
            device=device,
            output_path=output_path,
        )

    logger.info("Loading Textual Inversion model...")
    if textualinversion_model is not None:
        app.state.textualinversion_model = TextualInversion(
            model=textualinversion_model,
            device=device,
            output_path=output_path,
            token_identifier=ti_identifier,
            embeddings_url=ti_embeddings_url,
        )
    logger.info("Loading Canny model...")
    if canny_model is not None:
        app.state.canny_model = Canny(
            model=canny_model,
            device=device,
            output_path=output_path,
        )
    logger.info("Loading Openpose model...")
    if openpose_model is not None:
        app.state.openpose_model = Openpose(
            model=openpose_model,
            device=device,
            output_path=output_path,
        )
    logger.info("API is ready to use!")


@app.post("/text2img")
async def text2img(params: Text2ImgParams) -> ImgResponse:
    logger.info(f"Params: {params}")
    if app.state.x2img_model is None:
        return {"error": "x2img model is not loaded"}

    images, _ = app.state.x2img_model.text2img_generate(
        params.prompt,
        num_images=params.num_images,
        steps=params.steps,
        seed=params.seed,
        negative_prompt=params.negative_prompt,
        scheduler=params.scheduler,
        image_size=(params.image_height, params.image_width),
        guidance_scale=params.guidance_scale,
    )
    base64images = convert_to_b64_list(images)
    return ImgResponse(images=base64images, metadata=params.dict())


@app.post("/img2img")
async def img2img(params: Img2ImgParams = Depends(), image: UploadFile = File(...)) -> ImgResponse:
    if app.state.x2img_model is None:
        return {"error": "x2img model is not loaded"}
    image = Image.open(io.BytesIO(image.file.read()))
    images, _ = app.state.x2img_model.img2img_generate(
        image=image,
        prompt=params.prompt,
        negative_prompt=params.negative_prompt,
        num_images=params.num_images,
        steps=params.steps,
        seed=params.seed,
        scheduler=params.scheduler,
        guidance_scale=params.guidance_scale,
        strength=params.strength,
    )
    base64images = convert_to_b64_list(images)
    return ImgResponse(images=base64images, metadata=params.dict())

@app.post("/textualinversion")
async def textualinversion(params: TextualInversionParams = Depends(), image: UploadFile = File(...)) -> ImgResponse:
    logger.info(f"Params: {params}")
    if app.state.textualinversion_model is None:
        return {"error": "Textual inversion model is not loaded"}

    images, _ = app.state.textualinversion_model.generate_image(
        params.prompt,
        num_images=params.num_images,
        steps=params.steps,
        seed=params.seed,
        negative_prompt=params.negative_prompt,
        scheduler=params.scheduler,
        image_size=(params.image_height, params.image_width),

        guidance_scale=params.guidance_scale,
    )
    base64images = convert_to_b64_list(images)
    return ImgResponse(images=base64images, metadata=params.dict())


@app.post("/instruct-pix2pix")
async def instruct_pix2pix(params: InstructPix2PixParams = Depends(), image: UploadFile = File(...)) -> ImgResponse:
    if app.state.x2img_model is None:
        return {"error": "x2img model is not loaded"}
    image = Image.open(io.BytesIO(image.file.read()))
    images, _ = app.state.x2img_model.pix2pix_generate(
        image=image,
        prompt=params.prompt,
        negative_prompt=params.negative_prompt,
        num_images=params.num_images,
        steps=params.steps,
        seed=params.seed,
        scheduler=params.scheduler,
        guidance_scale=params.guidance_scale,
        image_guidance_scale=params.image_guidance_scale,
    )
    base64images = convert_to_b64_list(images)
    return ImgResponse(images=base64images, metadata=params.dict())


@app.post("/inpainting")
async def inpainting(
    params: InpaintingParams = Depends(), image: UploadFile = File(...), mask: UploadFile = File(...)
) -> ImgResponse:
    if app.state.inpainting_model is None:
        return {"error": "inpainting model is not loaded"}
    image = Image.open(io.BytesIO(image.file.read()))
    mask = Image.open(io.BytesIO(mask.file.read()))
    images, _ = app.state.inpainting_model.generate_image(
        image=image,
        mask=mask,
        prompt=params.prompt,
        negative_prompt=params.negative_prompt,
        scheduler=params.scheduler,
        height=params.image_height,
        width=params.image_width,
        num_images=params.num_images,
        guidance_scale=params.guidance_scale,
        steps=params.steps,
        seed=params.seed,
    )
    base64images = convert_to_b64_list(images)
    return ImgResponse(images=base64images, metadata=params.dict())

@app.post("/upscaler")
async def upscaler(
    params: UpscalerParams = Depends(), image: UploadFile = File(...)
) -> ImgResponse:
    if app.state.upscale_model is None:
        return {"error": "upscaler model is not loaded"}
    image = Image.open(io.BytesIO(image.file.read()))
    images, _ = app.state.upscale_model.generate_image(
        image=image,
        prompt=params.prompt,
        negative_prompt=params.negative_prompt,
        scheduler=params.scheduler,
        noise_level=params.noise_level,
        eta=params.eta,
        num_images=params.num_images,
        guidance_scale=params.guidance_scale,
        steps=params.steps,
        seed=params.seed,
    )
    base64images = convert_to_b64_list(images)
    return ImgResponse(images=base64images, metadata=params.dict())

@app.post("/canny")
async def canny(
    params: CannyParams = Depends(), image: UploadFile = File(...)
) -> ImgResponse:
    if app.state.canny_model is None:
        return {"error": "canny model is not loaded"}
    image = Image.open(io.BytesIO(image.file.read()))
    image = np.array(image)
    images, _ = app.state.canny_model.generate_image(
        image=image,
        prompt=params.prompt,
        negative_prompt=params.negative_prompt,
        low_threshold=params.low_threshold,
        high_threshold=params.high_threshold,
        num_images=params.num_images,
        guidance_scale=params.guidance_scale,
        steps=params.steps,
        seed=params.seed,
    )
    base64images = convert_to_b64_list(images)
    return ImgResponse(images=base64images, metadata=params.dict())

@app.post("/openpose")
async def openpose(
    params: OpenposeParams = Depends(), image: UploadFile = File(...)
) -> ImgResponse:
    if app.state.openpose_model is None:
        return {"error": "Openpose model is not loaded"}
    image = Image.open(io.BytesIO(image.file.read()))
    image = np.array(image)
    images, _ = app.state.openpose_model.generate_image(
        image=image,
        prompt=params.prompt,
        negative_prompt=params.negative_prompt,
        num_images=params.num_images,
        guidance_scale=params.guidance_scale,
        steps=params.steps,
        seed=params.seed,
    )
    base64images = convert_to_b64_list(images)
    return ImgResponse(images=base64images, metadata=params.dict())

@app.post("/image-info")
async def get_image_info(image: UploadFile = File(...)):
    # read image using PIL
    pil_image = Image.open(image.file)
    pil_image.verify()
    image_info = pil_image.info
    return {"image_info": image_info}

@app.get("/")
def read_root():
    return {"Hello": "World"}
