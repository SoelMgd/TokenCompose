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
        if attn.shape[-1] != 64 * 64:  # Vérifie la résolution
            return attn

        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        #print(f"Capturing attention map for key: {key}, shape: {attn.shape}")
        self.step_store[key].append(attn.clone())
        return attn

    def between_steps(self):
        #print(f"Resetting between steps at cur_step={self.cur_step}")
        if len(self.attention_store) > 0:
            #print(f"Attention store non vide ({len(self.attention_store)} clés). Réinitialisation forcée.")
            self.attention_store = {}

        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    

# Charger le modèle depuis HuggingFace
model_id = "mlpc-lab/TokenCompose_SD14_B"
device = "cuda"

print("Chargement de la pipeline Stable Diffusion...")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
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
attn_maps = get_cross_attn_map_from_unet(limited_attention_store, is_training_sd21=False)
print("Cartes d'attention extraites.")

print("Clés disponibles dans attention_maps :", limited_attention_store.attention_store.keys())
for key, maps in limited_attention_store.attention_store.items():
    print(f"Key: {key}")
    for i, map_ in enumerate(maps):
        print(f"  Map {i}: Shape {map_.shape}")



# Identifier les indices des tokens
print("Encodage des tokens pour le prompt...")
token_indices_cat = tokenizer.encode("cat")[1:-1]
token_indices_glass = tokenizer.encode("wine glass")[1:-1]
print(f"Indices pour 'cat': {token_indices_cat}")
print(f"Indices pour 'wine glass': {token_indices_glass}")
'''
# Visualiser et sauvegarder les cartes d'attention
print("Visualisation et sauvegarde des cartes d'attention...")
for token_name, token_indices in [("cat", token_indices_cat), ("wine_glass", token_indices_glass)]:
    try:
        # Vérifiez si la clé 'up_32' existe
        if "up_64" not in attn_maps:
            raise KeyError("up_64 n'existe pas dans attn_maps")
        
        # Moyenne des cartes d'attention pour toutes les têtes
        print(f"Traitement des cartes d'attention pour '{token_name}'...")
        attention_map_token = attn_maps["up_64"][0][..., token_indices].mean(dim=-1).cpu().numpy()
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
        '''


import torch.nn.functional as F

def reduce_attention_map(attention_map, target_resolution=512):
    # attention_map: torch.Size([16, 4096, 4096])
    attention_map = attention_map.mean(0)  # Moyenne sur les têtes d'attention (16 -> 1)
    attention_map = attention_map.view(4096, 4096, -1)  # Résolution brute

    # Réduction de la taille
    scale_factor = target_resolution / 4096
    reduced_map = F.interpolate(
        attention_map.permute(2, 0, 1).unsqueeze(0),  # Format (B, C, H, W)
        scale_factor=scale_factor,
        mode="bilinear"
    ).squeeze(0).permute(1, 2, 0)  # Retour au format original

    return reduced_map


def visualize_token_attention(attention_map, token_indices, target_resolution=512):
    # attention_map: torch.Size([4096, 4096, num_tokens])
    token_map = attention_map[..., token_indices].mean(dim=-1)  # Moyenne sur les indices des tokens
    reduced_map = reduce_attention_map(token_map, target_resolution=target_resolution)
    return reduced_map



# Afficher pour "cat"
if "up_cross" in limited_attention_store.attention_store:
    raw_attention_map = limited_attention_store.attention_store["up_cross"][0]  # Première carte brute
    attention_cat = visualize_token_attention(raw_attention_map, token_indices_cat, target_resolution=512)

    plt.figure(figsize=(8, 8))
    plt.imshow(attention_cat, cmap="viridis")
    plt.title("Cross-attention Map: 'cat'")
    plt.colorbar()
    plt.axis("off")
    plt.savefig("attention_map_cat.png")
    plt.show()

# Afficher pour "wine glass"
if "up_cross" in limited_attention_store.attention_store:
    attention_glass = visualize_token_attention(raw_attention_map, token_indices_glass, target_resolution=512)

    plt.figure(figsize=(8, 8))
    plt.imshow(attention_glass, cmap="viridis")
    plt.title("Cross-attention Map: 'wine glass'")
    plt.colorbar()
    plt.axis("off")
    plt.savefig("attention_map_wine_glass.png")
    plt.show()
