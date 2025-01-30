from transformers import pipeline
from PIL import Image
import requests

# load pipe
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

# load image
image = Image.open("assets/fence.jpg")

# inference
depth = pipe(image)["depth"]

img2 = Image.open("real/real.jpg")

depth2 = pipe(img2)["depth"]

depth2
