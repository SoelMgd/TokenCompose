import torch
from diffusers import StableDiffusionPipeline
from attn_utils import AttentionStore, register_attention_control, get_cross_attn_map_from_unet
from transformers import CLIPTokenizer
import matplotlib.pyplot as plt
import os

# Initialiser le répertoire de sortie
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Créer un contrôleur d'attention limité
class LimitedAttentionStore(AttentionStore):
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # Limiter uniquement à la couche 'up'
        if place_in_unet != "up":
            return attn
        
        # Limiter à la résolution 32
        if attn.shape[-1] != 32 * 32:  # Vérifie la résolution
            return attn
        
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.step_store[key].append(attn.clone())
        return attn
    

# Charger le modèle depuis HuggingFace
model_id = "mlpc-lab/TokenCompose_SD14_B"
device = "cuda"

print("Chargement de la pipeline Stable Diffusion...")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to(device)
print("Pipeline chargée avec succès.")

# Initialiser AttentionStore pour capturer les cartes d'attention
print("Initialisation d'AttentionStore...")
limited_attention_store  = LimitedAttentionStore()

# Remplacer le U-Net dans la pipeline pour enregistrer les cartes d'attention
print("Enregistrement d'AttentionStore dans le modèle U-Net...")
register_attention_control(pipe.unet, limited_attention_store )
print("AttentionStore enregistré avec succès.")

# Récupérer le tokenizer depuis la pipeline
print("Chargement du tokenizer...")
tokenizer = pipe.tokenizer
print("Tokenizer chargé :", type(tokenizer))

# Prompt pour l'inférence
prompt = "A cat and a wine glass"
print(f"Prompt : {prompt}")

# Générer une image
print("Début de la génération d'image...")
image = pipe(prompt).images[0]
print("Image générée avec succès.")

# Sauvegarder l'image générée
image_path = os.path.join(output_dir, "generated_image.png")
image.save(image_path)
print(f"Image sauvegardée à : {image_path}")

# Récupérer les cartes d'attention
print("Extraction des cartes de cross-attention...")
attn_maps = get_cross_attn_map_from_unet(attention_store, is_training_sd21=False)
print("Cartes d'attention extraites.")

# Identifier les indices des tokens
print("Encodage des tokens pour le prompt...")
token_indices_cat = tokenizer.encode("cat")[1:-1]
token_indices_glass = tokenizer.encode("wine glass")[1:-1]
print(f"Indices pour 'cat': {token_indices_cat}")
print(f"Indices pour 'wine glass': {token_indices_glass}")

# Visualiser et sauvegarder les cartes d'attention
print("Visualisation et sauvegarde des cartes d'attention...")
for token_name, token_indices in [("cat", token_indices_cat), ("wine_glass", token_indices_glass)]:
    try:
        # Moyenne des cartes d'attention pour toutes les têtes
        print(f"Traitement des cartes d'attention pour '{token_name}'...")
        attention_map_token = attn_maps["down_64"][0][..., token_indices].mean(dim=-1).cpu().numpy()
        print(f"Carte d'attention pour '{token_name}' extraite avec succès.")

        # Sauvegarder la carte d'attention
        plt.figure(figsize=(8, 8))
        plt.imshow(attention_map_token, cmap="viridis")
        plt.title(f"Cross-attention Map: {token_name}")
        plt.colorbar()
        plt.axis("off")
        attention_map_path = os.path.join(output_dir, f"attention_map_{token_name}.png")
        plt.savefig(attention_map_path)
        plt.close()
        print(f"Carte d'attention pour '{token_name}' sauvegardée à : {attention_map_path}")
    except Exception as e:
        print(f"Erreur lors du traitement de '{token_name}': {e}")
