# Install required libraries
!pip install -q -U accelerate bitsandbytes transformers

# Import necessary libraries and login to Hugging Face
from huggingface_hub import notebook_login
notebook_login()

import torch
from transformers import AutoTokenizer, PaliGemmaForConditionalGeneration, PaliGemmaProcessor

# Set up device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model.to(device)
processor = PaliGemmaProcessor.from_pretrained(model_id)

# Function to load and display an image
import torch
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt

def load_image(url):
    input_image = Image.open(requests.get(url, stream=True).raw)
    plt.imshow(input_image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()
    return input_image

# Load an example image
peacock_picture = load_image("https://static.vecteezy.com/system/resources/previews/032/257/185/non_2x/wallpapers-for-the-beautiful-peacock-wallpaper-ai-generated-free-photo.jpg")

# Function to query the model
def query(image, prompt):
    inputs = processor(text=prompt, images=image,
                       padding="longest", do_convert_rgb=True, return_tensors="pt").to("cuda")
    inputs = inputs.to(dtype=model.dtype)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=496)
    return processor.decode(output[0], skip_special_tokens=True)

# Function to display bounding box
import re
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def display_bounding_box(pil_image, model_output):
    # Parse the location tokens
    loc_values = re.findall(r'<loc(\d+)>', model_output)
    loc_values = [int(val) for val in loc_values]
    
    # Convert normalized coordinates to image coordinates
    width, height = pil_image.size
    y_min, x_min, y_max, x_max = [
        int(loc_values[i] / 1024 * (height if i % 2 == 0 else width))
        for i in range(4)
    ]
    
    # Create a copy of the image to draw on
    draw_image = pil_image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Draw the bounding box
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(draw_image)
    plt.axis('off')
    plt.show()

# Example usage
output = query(peacock_picture, "detect peacock")
print(output)
display_bounding_box(peacock_picture, output)
