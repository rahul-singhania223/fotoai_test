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

def upscale_image(image, iters=1):
    model = RealESRGANModel(2)
    result = None
    for _ in range(iters):
        result = model.process_from_image(image)

    return result

def fix_light(image):
    model = DCENetModel()
    return model.process_from_image(image, alpha=0.3)




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
        obj_upscaled = self.upscale_2x(obj_img, iters=2)
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

        img = obj_light_fixed.resize((int(imgW), int(imgH)))

        background = Image.new('RGB', (w, h), (255, 255, 255))
        background.paste(img, (paste_x, paste_y), mask=img.split()[3])

        final_image = background

        print("Resized image complete")

        # upload
        format = settings['format']
        buffer = BytesIO()
        final_image.save(buffer, format=format.upper())
        obj_buffer = buffer.getvalue()

        upload_res = upload_to_supabase(obj_buffer, format)

        return upload_res
        

        

