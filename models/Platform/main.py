from models.RealESRGAN.main import RealESRGANModel
from models.BiRefNet.main import BiRefNetModel
from models.Zero_DCE.main import DCENetModel

from lib.utils import upload_to_supabase

from PIL import Image
import requests
from io import BytesIO

class PlatformModel:
    def __init__(self):
        self.birefnet_model = BiRefNetModel()
        self.realesrgan_model = RealESRGANModel(4)
        self.dce_model = DCENetModel()

    def process(self, image_url, settings={ 'dimension': 1024, 'format': 'png'}):     
        res = requests.get(image_url)
        image = Image.open(BytesIO(res.content))
        image = image.convert('RGB')

        # upspscale <= dimension
        img_upscaled = self.realesrgan_model.process_from_image(image)
        
        # extract object
        obj_img = self.birefnet_model.extract_object_from_image(img_upscaled)

        # light fix
        obj_light_fixed = self.dce_model.process_from_image(obj_img, alpha=0.5)

                
        # upload
        format = settings['format']
        buffer = BytesIO()
        obj_light_fixed.save(buffer, format=format.upper())
        obj_buffer = buffer.getvalue()

        upload_res = upload_to_supabase(obj_buffer, format)

        return upload_res
        

        

