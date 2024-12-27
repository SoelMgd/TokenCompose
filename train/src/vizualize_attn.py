import torch
from diffusers import StableDiffusionPipeline
from attn_utils import AttentionStore, register_attention_control, get_cross_attn_map_from_unet
import os
import matplotlib.pyplot as plt

model_id = "mlpc-lab/TokenCompose_SD14_B"
device = "cuda"

# Initialiser AttentionStore
attention_store = AttentionStore()

# Charger la pipeline avec FP16
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
pipe.to(device)

# Enregistrer AttentionStore dans le U-Net
register_attention_control(pipe.unet, attention_store)

prompt = "A cat and a wine glass"

# Générer l'image avec torch.no_grad()
with torch.no_grad():
    image = pipe(prompt).images[0]

# Libérer la mémoire GPU utilisée par l'image
#del image
#torch.cuda.empty_cache()

# Récupérer les cartes d'attention (résolutions réduites)
attn_maps = get_cross_attn_map_from_unet(attention_store, is_training_sd21=False, reses=[16, 8])

# Sauvegarder la carte d'attention
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
attention_map = attn_maps["down_16"][0].mean(dim=-1).cpu().numpy()

plt.figure(figsize=(8, 8))
plt.imshow(attention_map, cmap="viridis")
plt.title("Cross-attention Map")
plt.colorbar()
plt.savefig(os.path.join(output_dir, "attention_map.png"))
plt.close()
