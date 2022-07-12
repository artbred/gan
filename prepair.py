import os

from PIL import Image

base_path = "LLD-logo_sample"
output = "dataset"

logos = os.listdir(base_path)

for logo in logos:
    im = Image.open(f'{base_path}/{logo}')
    rgb_im = im.convert('RGB')
    rgb_im.save(f'{output}/{logo.replace("png", "jpg")}')