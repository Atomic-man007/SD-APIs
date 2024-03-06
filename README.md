# SD-APIs

stable diffusion APIs

APIs List:-
canny
depth estimator
image segmentation
inpainting
openpose
textual inversion
upscale

---------------------
![Text2Img](/data/text2img/2023-06-28_18-21-39/0.png)

```
pip install -r requirements.txt
```

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

```
python main.py
```

```
uvicorn main:app --reload
```

```
pip install -U --pre triton
```
