import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from attn_utils import AttentionStore, register_attention_control
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

        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.step_store[key].append(attn.clone())
        return attn

    def between_steps(self):
        if len(self.attention_store) > 0:
            self.attention_store = {}

        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()


# Fonction pour réduire les cartes d'attention à une résolution visualisable
def reduce_attention_map(attention_map, target_resolution=512):
    # Moyenne sur les têtes d'attention (16 -> 1)
    attention_map = attention_map.mean(0)
    # Réduction à la résolution cible
    scale_factor = target_resolution / int(attention_map.shape[0] ** 0.5)
    reduced_map = F.interpolate(
        attention_map.view(1, 1, int(attention_map.shape[0] ** 0.5), -1),
        scale_factor=scale_factor,
        mode="bilinear"
    ).squeeze(0)
    return reduced_map.squeeze(0).cpu().numpy()


# Charger le modèle depuis HuggingFace
model_id = "mlpc-lab/TokenCompose_SD14_B"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.to(device)
print("Pipeline chargée avec succès.")

# Initialiser AttentionStore pour capturer les cartes d'attention
print("Initialisation d'AttentionStore...")
limited_attention_store = LimitedAttentionStore()

# Remplacer le U-Net dans la pipeline pour enregistrer les cartes d'attention
register_attention_control(pipe.unet, limited_attention_store)
print("AttentionStore enregistré avec succès.")

tokenizer = pipe.tokenizer
print("Tokenizer chargé :", type(tokenizer))

prompt = "A cat and a wine glass"

# Générer une image
image = pipe(prompt).images[0]
print("Image générée avec succès.")

# Sauvegarder l'image générée
image_path = os.path.join(output_dir, "generated_image.png")
image.save(image_path)
print(f"Image sauvegardée à : {image_path}")

# Récupérer les cartes d'attention
print("Clés disponibles dans attention_maps :", limited_attention_store.attention_store.keys())
attn_key = "up_cross"  # Choisir une clé pertinente (par exemple, up_cross)
if attn_key not in limited_attention_store.attention_store:
    raise ValueError(f"Clé {attn_key} non trouvée dans attention_maps.")

raw_attention_map = limited_attention_store.attention_store[attn_key][0]  # Première carte brute
print(f"Dimensions de la carte brute : {raw_attention_map.shape}")

# Moyenne sur les têtes d'attention
attention_map_avg = raw_attention_map.mean(dim=0)  # torch.Size([256, 77])
print(f"Dimensions après moyenne des têtes : {attention_map_avg.shape}")

# Interpolation pour une meilleure résolution
def upscale_attention_map(attention_map, target_resolution=512):
    return F.interpolate(
        attention_map.view(1, 1, 16, 16),  # Reshape vers (B, C, H, W)
        size=(target_resolution, target_resolution),
        mode="bilinear",
        align_corners=False
    ).squeeze(0).squeeze(0)  # Retour à (H, W)

# Sauvegarder les cartes d'attention pour tous les tokens
for token_idx in range(attention_map_avg.shape[-1]):  # Pour chaque token
    try:
        # Extraire la carte d'attention pour le token
        token_attention_map = attention_map_avg[:, token_idx]  # torch.Size([256])
        token_attention_map = upscale_attention_map(token_attention_map)  # torch.Size([512, 512])

        # Sauvegarder l'image
        plt.figure(figsize=(8, 8))
        plt.imshow(token_attention_map.cpu().numpy(), cmap="viridis")
        plt.title(f"Cross-attention Map: Token {token_idx}")
        plt.colorbar()
        plt.axis("off")
        attention_map_path = os.path.join(output_dir, f"attention_map_token_{token_idx}.png")
        plt.savefig(attention_map_path)
        plt.close()
        print(f"Carte d'attention pour le token {token_idx} sauvegardée à : {attention_map_path}")
    except Exception as e:
        print(f"Erreur lors du traitement du token {token_idx}: {e}")