from models.RealESRGAN.main import RealESRGANModel
from models.BiRefNet.main import BiRefNetModel
from models.Zero_DCE.main import DCENetModel

from lib.utils import upload_to_supabase

from PIL import Image
import requests
from io import BytesIO

def extract_object(image):
    model = BiRefNetModel()
    return model.extract_object_from_image(image)

def upscale_image(image):
    model = RealESRGANModel(2)
    return model.process_from_image(image)

def fix_light(image):
    model = DCENetModel()
    return model.process_from_image(image)




class PlatformModel:
    def __init__(self):
        self.extract_object = extract_object
        self.upscale_2x = upscale_image
        self.light_fix = fix_light

    def process(self, image_url, settings={ 'dimension': 2000, 'format': 'png'}):     
        res = requests.get(image_url)
        image = Image.open(BytesIO(res.content))
        

    
        # Etract object
        print("Extracting object...")
        obj_img = self.extract_object(image)
        print("Extracted object complete")

        # Upscale
        print("Upscaling object...")
        obj_upscaled = self.upscale_2x(obj_img)
        print("Upscaled object complete")

        # Upscale
        print("Upscaling object again...")
        obj_upscaled = self.upscale_2x(obj_img)
        print("Upscaled object complete")

        # Light fix
        print("Fixing light...")
        obj_light_fixed = self.light_fix(obj_upscaled)
        print("Fixed light complete")

        # Resize
        print("Resizing image...")
        w, h = settings['dimension'], settings['dimension']
        imgW = w * 0.85
        imgH = imgW/obj_light_fixed.width * obj_light_fixed.height
        paste_x = (w - int(imgW)) // 2
        paste_y = (h - int(imgH)) // 2

        background = Image.new('RGB', (w, h), (255, 255, 255))
        background.paste(obj_light_fixed, (paste_x, paste_y), mask=obj_light_fixed.split()[3])

        final_image = background

        print("Resized image complete")

        # upload
        format = settings['format']
        buffer = BytesIO()
        final_image.save(buffer, format=format.upper())
        obj_buffer = buffer.getvalue()

        upload_res = upload_to_supabase(obj_buffer, format)

        return upload_res
        

        

