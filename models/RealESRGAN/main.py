import torch
from PIL import Image
import requests
from io import BytesIO

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

