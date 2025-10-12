import torch
from PIL import Image
import requests
from io import BytesIO
import numpy as np

from . import RealESRGAN
from lib.utils import upload_to_supabase


class RealESRGANModel:
    def __init__(self, scale=2):
        device = torch.device('cuda')
        self.model = RealESRGAN(device, scale)
        self.model.load_weights(f'models/RealESRGAN/weights/RealESRGAN_x{scale}.pth')

    def process_from_image(self, image):
        # prediction
        print("Upscaling image...")
        result_image = self.model.predict(image)
        
        return result_image
        

    def process(self, image_url, settings={}):
        # download image
        print("Downloading image...")
        res = requests.get(image_url)
        image = Image.open(BytesIO(res.content))

        n_channels = np.split()[-1]
        if n_channels == 4:
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background

        # prediction
        print("Upscaling image...")
        result_image = self.model.predict(image)

        # upload to storage
        print("Uploading image...")
        format = res.headers.get('Content-Type').split('/')[-1]
        buffer = BytesIO()
        result_image.save(buffer, format=format.upper())
        image_buffer = buffer.getvalue()

        upload_result = upload_to_supabase(image_buffer, format)
        print("Process complete.")
        return upload_result

