import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import requests
from io import BytesIO

from models.Zero_DCE.model import DCENet
from lib.utils import upload_to_supabase

class DCENetModel:
    def __init__(self):
        # Load model

        model_path = "models/Zero_DCE/weights/Epoch99.pth"
        self.device = "cuda"

        self.model = DCENet().cuda()
        self.model.load_state_dict(torch.load(model_path, weights_only=False))

    def process_from_image(self, raw_image, alpha=1.0):
        img = raw_image.convert('RGB')
        img = (np.asarray(img)/255.0)
        img = torch.from_numpy(img).float()
        img = img.permute(2,0,1)
        img = img.cuda().unsqueeze(0)

        # forward pass
        with torch.no_grad():
            _, enhanced_image, _ = self.model(img, alpha=alpha)

        img = enhanced_image.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)  # [H, W, C]
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(img)

        return img



    def process(self, image_url, settings={'alpha': 1.0}):
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        alpha = settings['alpha']
        # load image
        res = requests.get(image_url)
        image = Image.open(BytesIO(res.content)).convert('RGB')
        image = (np.asarray(image)/255.0)
        image = torch.from_numpy(image).float()
        image = image.permute(2,0,1)
        image = image.cuda().unsqueeze(0)
       
        # forward pass
        with torch.no_grad():
            _, enhanced_image, _ = self.model(image, alpha=alpha)

        img = enhanced_image.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)  # [H, W, C]
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(img)

        # upload
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_buffer = buffer.getvalue()    
        format = res.headers.get('Content-Type').split('/')[-1]


        upload_res = upload_to_supabase(image_buffer, format)
        return upload_res



