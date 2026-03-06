import gradio as gr
from PIL import  Image
from transformers import AutoProcessor,BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image):

    raw_image = input_image.convert('RGB')

    inputs = processor(images=raw_image,return_tensors="pt")

    out = model.generate(**inputs,max_length=50)

    caption = processor.decode(out[0],skip_special_tokens=True)

    return caption

iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Raghav's Local AI vision App",
    description="Drag and drop any image here,and offline BLIP model tell you what is sees!"
)

if __name__ == "__main__":
    iface.launch()


