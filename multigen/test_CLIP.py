import torch
from diffusers import StableDiffusionPipeline

model_id = "mlpc-lab/TokenCompose_SD14_A"
device = "cuda"

# Charger la pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to(device)

# Inspecter le modèle CLIP utilisé (encodeur texte)
text_encoder = pipe.text_encoder

# Afficher des informations sur l'architecture
print(f"CLIP Text Encoder: {text_encoder.config.architectures}")
print(f"Number of layers: {text_encoder.config.num_hidden_layers}")
print(f"Hidden size: {text_encoder.config.hidden_size}")
