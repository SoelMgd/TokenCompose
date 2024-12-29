import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

model_id = "mlpc-lab/TokenCompose_SD21_A"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# Charger un autre modèle CLIP
new_clip_model_id = "openai/clip-vit-base-patch32"  # Ou votre modèle fine-tuné
new_text_encoder = CLIPTextModel.from_pretrained(new_clip_model_id)
new_tokenizer = CLIPTokenizer.from_pretrained(new_clip_model_id)

# Remplacer les composants CLIP dans la pipeline
pipe.text_encoder = new_text_encoder
pipe.tokenizer = new_tokenizer

pipe = pipe.to(device)

prompt = "A cat and a wine glass"
image = pipe(prompt).images[0]  

image.save('testing_image.png')
