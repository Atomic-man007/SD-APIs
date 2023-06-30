import base64
import gc
import io
import os
import tempfile
import zipfile
from datetime import datetime
from threading import Thread

import requests
import streamlit as st
import torch
from huggingface_hub import HfApi
from huggingface_hub.utils._errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from loguru import logger
from PIL.PngImagePlugin import PngInfo

no_safety_checker = None

CODE_OF_CONDUCT = """
## Code of conduct
The app should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

Using the app to generate content that is cruel to individuals is a misuse of this app. One shall not use this app to generate content that is intended to be cruel to individuals, or to generate content that is intended to be cruel to individuals in a way that is not obvious to the viewer.
This includes, but is not limited to:
- Generating demeaning, dehumanizing, or otherwise harmful representations of people or their environments, cultures, religions, etc.
- Intentionally promoting or propagating discriminatory content or harmful stereotypes.
- Impersonating individuals without their consent.
- Sexual content without consent of the people who might see it.
- Mis- and disinformation
- Representations of egregious violence and gore
- Sharing of copyrighted or licensed material in violation of its terms of use.
- Sharing content that is an alteration of copyrighted or licensed material in violation of its terms of use.

By using this app, you agree to the above code of conduct.

"""


def use_auth_token():
    token_path = os.path.join(os.path.expanduser("~"), ".huggingface", "token")
    if os.path.exists(token_path):
        return True
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]
    return False


def download_file(file_url):
    r = requests.get(file_url, stream=True)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                tmp.write(chunk)
    return tmp.name


def cache_folder():
    _cache_folder = os.path.join(os.path.expanduser("~"), ".diffuzers")
    os.makedirs(_cache_folder, exist_ok=True)
    return _cache_folder


def clear_memory(preserve):
    torch.cuda.empty_cache()
    gc.collect()
    to_clear = ["inpainting", "text2img", "img2text"]
    for key in to_clear:
        if key not in preserve and key in st.session_state:
            del st.session_state[key]


def save_to_hub(api, images, module, current_datetime, metadata, output_path):
    logger.info(f"Saving images to hub: {output_path}")
    _metadata = PngInfo()
    _metadata.add_text("text2img", metadata)
    for i, img in enumerate(images):
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG", pnginfo=_metadata)
        img_byte_arr = img_byte_arr.getvalue()
        api.upload_file(
            path_or_fileobj=img_byte_arr,
            path_in_repo=f"{module}/{current_datetime}/{i}.png",
            repo_id=output_path,
            repo_type="dataset",
        )

    api.upload_file(
        path_or_fileobj=str.encode(metadata),
        path_in_repo=f"{module}/{current_datetime}/metadata.json",
        repo_id=output_path,
        repo_type="dataset",
    )
    logger.info(f"Saved images to hub: {output_path}")


def save_to_local(images, module, current_datetime, metadata, output_path):
    _metadata = PngInfo()
    _metadata.add_text("text2img", metadata)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/{module}", exist_ok=True)
    os.makedirs(f"{output_path}/{module}/{current_datetime}", exist_ok=True)

    for i, img in enumerate(images):
        img.save(
            f"{output_path}/{module}/{current_datetime}/{i}.png",
            pnginfo=_metadata,
        )

    # save metadata as text file
    with open(f"{output_path}/{module}/{current_datetime}/metadata.txt", "w") as f:
        f.write(metadata)
    logger.info(f"Saved images to {output_path}/{module}/{current_datetime}")

def save_to_local_controlnet(images,poses, module, current_datetime, metadata, output_path):
    _metadata = PngInfo()
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/{module}", exist_ok=True)
    os.makedirs(f"{output_path}/{module}/{current_datetime}", exist_ok=True)

    for i, img in enumerate(images):
        img.save(
            f"{output_path}/{module}/{current_datetime}/{i}.png",
            pnginfo=_metadata,
        )
    poses.save(
        f"{output_path}/{module}/{current_datetime}/{i}_pose.png",
    )

    # save metadata as text file
    with open(f"{output_path}/{module}/{current_datetime}/metadata.txt", "w") as f:
        f.write(metadata)
    logger.info(f"Saved images to {output_path}/{module}/{current_datetime}")


def save_images(images, module, metadata, output_path):
    if output_path is None:
        logger.warning("No output path specified, skipping saving images")
        return

    api = HfApi()
    dset_info = None
    try:
        dset_info = api.dataset_info(output_path)
    except (HFValidationError, RepositoryNotFoundError):
        logger.warning("No valid hugging face repo. Saving locally...")

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not dset_info:
        save_to_local(images, module, current_datetime, metadata, output_path)
    else:
        Thread(target=save_to_hub, args=(api, images, module, current_datetime, metadata, output_path)).start()
