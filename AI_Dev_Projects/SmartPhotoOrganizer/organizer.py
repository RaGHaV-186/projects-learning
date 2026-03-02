import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def organize_photos():
    # 1. Initialize the processor and model from Hugging Face
    # The first time you run this, it will download the model files to your Mac
    print("Loading the BLIP AI model... (This might take a minute the first time)")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # 2. Define the folder where your images are stored
    image_folder = "raw_images"

    # 3. Loop through all files in the folder
    print(f"Scanning the '{image_folder}' folder...")
    for filename in os.listdir(image_folder):
        # Only process image files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {filename}")
            image_path = os.path.join(image_folder, filename)

            # Open the image using Pillow and convert to standard RGB
            raw_image = Image.open(image_path).convert('RGB')

            # Prepare the image for the model
            inputs = processor(raw_image, return_tensors="pt")

            # Generate the text caption
            outputs = model.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)

            print(f"AI saw: '{caption}'")

            # Format the caption to be a valid file name (replace spaces with underscores)
            clean_caption = caption.replace(" ", "_").replace("/", "")

            # Get the original file extension (like .jpg)
            file_ext = os.path.splitext(filename)[1]

            # Create the new file name
            new_filename = f"{clean_caption}{file_ext}"
            new_filepath = os.path.join(image_folder, new_filename)

            # Rename the file on your hard drive
            os.rename(image_path, new_filepath)
            print(f"Success! Renamed to: {new_filename}\n")

    print("Finished organizing photos!")


if __name__ == '__main__':
    organize_photos()