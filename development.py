from models.RealESRGAN.main import RealESRGANModel
from models.BiRefNet.main import BiRefNetModel
from models.Zero_DCE.main import DCENetModel
from models.Platform.main import PlatformModel

import argparse
import json

processor_dict = {
    "REMOVE_BG": BiRefNetModel(),
    "UPSCALE_2X": RealESRGANModel(scale=2),
    "UPSCALE_4X": RealESRGANModel(scale=4),
    "LIGHT_FIX": DCENetModel(),
    "PLATFORM": PlatformModel() 
}

process_with_settings = ['LIGHT_FIX', 'PLATFORM']

def process_image(input):
    
    job_input = input['input']
    image_url = job_input['image_url']
    process_type = job_input['process_type']  

    agent = processor_dict[process_type]
    settings = {}

    # process with settings
    if process_type in process_with_settings:
        settings = job_input['settings']  

    output = agent.process(image_url=image_url, settings=settings)
    return output

def main():
    # accept arguments
        parser = argparse.ArgumentParser(description="Select processor.")        
        parser.add_argument("--input", type=str, required=True)
        args = parser.parse_args()

        with open(args.input, 'r') as f:
            input_data = json.load(f)

            output = process_image(input_data)

            with open(args.output, 'w') as f:
                json.dump(output, f)



if __name__ == "__main__":
    main()


