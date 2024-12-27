import torch
from diffusers import StableDiffusionPipeline
from attn_utils import AttentionStore, register_attention_control, get_cross_attn_map_from_unet
from transformers import CLIPTokenizer
import matplotlib.pyplot as plt
import os

# Charger le modèle depuis HuggingFace
model_id = "mlpc-lab/TokenCompose_SD14_A"
device = "cuda"

# Initialiser AttentionStore pour capturer les cartes d'attention
attention_store = AttentionStore()

# Charger le pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.to(device)

# Remplacer le U-Net dans la pipeline pour enregistrer les cartes d'attention
register_attention_control(pipe.unet, attention_store)

# Tokenizer pour identifier les indices des tokens
tokenizer = CLIPTokenizer.from_pretrained(model_id)

# Prompt
prompt = "A cat and a wine glass"

# Générer une image
image = pipe(prompt).images[0]

# Sauvegarder l'image générée
output_dir = "./viz_attn"
os.makedirs(output_dir, exist_ok=True)
image.save(os.path.join(output_dir, "generated_image.png"))

# Récupérer les cartes d'attention
attn_maps = get_cross_attn_map_from_unet(attention_store, is_training_sd21=False)

# Trouver l'index des tokens
tokens = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True)
token_indices_cat = tokenizer.encode("cat")[1:-1]  # Indices pour "cat"
token_indices_glass = tokenizer.encode("wine glass")[1:-1]  # Indices pour "wine glass"

# Visualiser et sauvegarder les cartes d'attention
for token_name, token_indices in [("cat", token_indices_cat), ("wine_glass", token_indices_glass)]:
    # Moyenne des cartes d'attention pour toutes les têtes, pour le premier niveau "down_64"
    attention_map_token = attn_maps["down_64"][0][..., token_indices].mean(dim=-1).cpu().numpy()

    # Sauvegarder la carte d'attention sous forme d'image
    plt.figure(figsize=(8, 8))
    plt.imshow(attention_map_token, cmap="viridis")
    plt.title(f"Cross-attention Map: {token_name}")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, f"attention_map_{token_name}.png"))
    plt.close()
