from PIL import Image
import numpy as np

def inspect_mask(mask_path):
    # Charger le masque
    mask = Image.open(mask_path)
    mask_np = np.array(mask)

    # Afficher des informations générales
    print(f"Processing mask: {mask_path}")
    print(f"Mask shape: {mask_np.shape}")
    print(f"Unique values in mask: {np.unique(mask_np)}")

    # Vérifier si le masque a plusieurs canaux
    if mask_np.ndim > 1:
        print(f"Mask has {mask_np.shape[2]} channels.")
        for channel in range(mask_np.shape[2]):
            print(f"Channel {channel}: Non-zero values = {np.count_nonzero(mask_np[..., channel])}")
            print(f"Unique values in channel {channel}: {np.unique(mask_np[..., channel])}")
    else:
        print("Mask is single-channel.")
        print(f"Non-zero values = {np.count_nonzero(mask_np)}")
        print(f"Unique values: {np.unique(mask_np)}")

# Exemple d'utilisation
mask_path = "data/coco_gsam_seg/000000053289/mask_000000053289_couch.png"  # Chemin d'un masque
inspect_mask(mask_path)
