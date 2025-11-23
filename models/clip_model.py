from PIL.ImageFile import ImageFile
import torch
from transformers import CLIPProcessor, CLIPVisionModel

class ClipModel():
    def __init__(self):
        vision_model = "openai/clip-vit-base-patch32"
        self.vision = CLIPVisionModel.from_pretrained(vision_model, device_map="auto")
        self.processor = CLIPProcessor.from_pretrained(vision_model, use_fast=True, device_map="auto")


    def get_model_size(self):
        vision_config = self.vision.config.get_text_config()
        vision_size = vision_config.hidden_size

        if vision_size > 0:
            return vision_size
        raise Exception("NO VISION MODEL FOUND")

    def get_image_ids(self, image: ImageFile):
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.vision.device) # type: ignore

        with torch.no_grad():
            output = self.vision(**inputs)

            image_feature : torch.Tensor = output.last_hidden_state

        return image_feature

