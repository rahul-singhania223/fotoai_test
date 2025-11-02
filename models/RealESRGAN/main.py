import torch
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import cv2

from . import RealESRGAN
from lib.utils import upload_to_supabase



class RealESRGANModel:
    def __init__(self, scale=2):
        device = torch.device('cuda')
        self.model = RealESRGAN(device, scale)
        self.model.load_weights(f'models/RealESRGAN/weights/RealESRGAN_x{scale}.pth')

    def process_from_image(self, image):      
        final_image = None

        if image.mode == 'RGBA':
            rgb_img = image.convert('RGB')
            alpha = np.array(image.getchannel('A'))

            rgb_up = self.model.predict(rgb_img)
            rgb_up_np = np.array(rgb_up)

            h, w = rgb_up_np.shape[:2]
            alpha_up = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_CUBIC)

            final_image = Image.fromarray(np.dstack((rgb_up_np, alpha_up)), 'RGBA')

        else:
            final_image = self.model.predict(image)
        
        return final_image
        

    def process(self, image_url, settings=None):
        # download image
        print("Downloading image...")
        res = requests.get(image_url)
        image = Image.open(BytesIO(res.content))

        print("Upscaling image...")

        final_image = None

        if image.mode == 'RGBA':
            rgb_img = image.convert('RGB')
            alpha = np.array(image.getchannel('A'))

            rgb_up = self.model.predict(rgb_img)
            rgb_up_np = np.array(rgb_up)

            h, w = rgb_up_np.shape[:2]
            alpha_up = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_CUBIC)

            final_image = Image.fromarray(np.dstack((rgb_up_np, alpha_up)), 'RGBA')

        else:
            final_image = self.model.predict(image)

        # upload to storage
        print("Uploading image...")
        format = res.headers.get('Content-Type').split('/')[-1]
        buffer = BytesIO()
        final_image.save(buffer, format=format.upper())
        image_buffer = buffer.getvalue()

        upload_result = upload_to_supabase(image_buffer, format)
        print("Process complete.")
        return upload_result

