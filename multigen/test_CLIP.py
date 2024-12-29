import torch
from torch import nn
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer



# Charger la pipeline Stable Diffusion
model_id = "mlpc-lab/TokenCompose_SD14_A"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# Charger le modèle CLIP avec une taille de sortie de 512
new_clip_model_id = "openai/clip-vit-base-patch16" #"openai/clip-vit-base-patch32"
new_text_encoder = CLIPTextModel.from_pretrained(new_clip_model_id)
new_tokenizer = CLIPTokenizer.from_pretrained(new_clip_model_id)


# Remplacer les composants CLIP dans la pipeline
pipe.text_encoder = new_text_encoder
pipe.tokenizer = new_tokenizer

# Transférer la pipeline sur le GPU
pipe = pipe.to(device)

# Effectuer une inférence
prompt = "A cat and a wine glass"
image = pipe(prompt).images[0]

# Sauvegarder l'image
image.save("testing_image.png")
