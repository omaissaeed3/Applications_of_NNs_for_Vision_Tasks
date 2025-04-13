from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

# Load a ViT-based captioning model with attention
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Preprocess image
image_path = "./Puppy2.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")
caption_ids = model.generate(**inputs)

# Decode caption
caption = tokenizer.decode(caption_ids[0], skip_special_tokens=True)
print(f"Generated Caption with Attention: {caption}")