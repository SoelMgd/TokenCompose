import torch
from diffusers import StableDiffusionPipeline
from attn_utils import AttentionStore, register_attention_control, get_cross_attn_map_from_unet
from transformers import CLIPTokenizer
import matplotlib.pyplot as plt
import os

# Charger le modèle depuis HuggingFace
model_id = "mlpc-lab/TokenCompose_SD14_B"
device = "cuda"

# Initialiser AttentionStore pour capturer les cartes d'attention
attention_store = AttentionStore()

print("chargement de la pipeline")
# Charger la pipeline avec précision mixte pour réduire la charge mémoire
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to(device)

print("pipeline chargee")

# Remplacer le U-Net dans la pipeline pour enregistrer les cartes d'attention
register_attention_control(pipe.unet, attention_store)
print("Unet remplacé")

# Récupérer le tokenizer depuis la pipeline
tokenizer = pipe.tokenizer

# Prompt
prompt = "A cat and a wine glass"

# Désactiver les gradients pour économiser de la mémoire
with torch.no_grad():
    # Générer une image
    print("génération de l'image")
    image = pipe(prompt).images[0]

    print("sauvegarde")
    # Sauvegarder l'image générée
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    image.save(os.path.join(output_dir, "generated_image.png"))

    # Extraire les cartes de cross-attention uniquement pour la couche "down_64"
    printt("extraction attention maps")
    attn_maps = get_cross_attn_map_from_unet(attention_store, is_training_sd21=False)

    # Identifier l'index d'un seul token (par exemple, "cat")
    token_indices_cat = tokenizer.encode("cat")[1:-1]  # Indices pour "cat"

    # Moyenne des cartes d'attention pour toutes les têtes à "down_64" pour le token "cat"
    attention_map_token = attn_maps["down_64"][0][..., token_indices_cat].mean(dim=-1).cpu().numpy()

    # Sauvegarder la carte d'attention
    plt.figure(figsize=(8, 8))
    plt.imshow(attention_map_token, cmap="viridis")
    plt.title(f"Cross-attention Map: cat")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, f"attention_map_cat.png"))
    plt.close()
