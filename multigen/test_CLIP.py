import torch
from torch import nn
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer



# Charger la pipeline Stable Diffusion
model_id = "mlpc-lab/TokenCompose_SD14_A"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

'''
# Charger le modèle CLIP
new_clip_model_id = "openai/clip-vit-large-patch14-336" #"openai/clip-vit-base-patch32"
new_text_encoder = CLIPTextModel.from_pretrained(new_clip_model_id)
new_tokenizer = CLIPTokenizer.from_pretrained(new_clip_model_id)'''

#fine_tuned_clip_model_dir = "../train/clip_finetuned_model"  # Chemin où vos poids sont sauvegardés
#new_text_encoder = CLIPTextModel.from_pretrained(fine_tuned_clip_model_dir)
#new_tokenizer = CLIPTokenizer.from_pretrained(fine_tuned_clip_model_dir)


# Remplacer les composants CLIP dans la pipeline
#pipe.text_encoder = new_text_encoder
#pipe.tokenizer = new_tokenizer

# Transférer la pipeline sur le GPU
pipe = pipe.to(device)

# Effectuer une inférence
prompt = "A toilet in front of the ocean"
image = pipe(prompt).images[0]

# Sauvegarder l'image
image.save("testing_image.png")
