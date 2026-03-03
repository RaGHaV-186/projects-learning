import gradio as gd
from PIL import  Image
from transformers import AutoProcessor,BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image):

    raw_image = input_image.convert('RGB')

    inputs = processor(images=raw_image,return_tensors="pt")

    out = model.generate(**inputs,max_length=50)