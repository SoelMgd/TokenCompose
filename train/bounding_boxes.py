import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_bounding_boxes(img_path, mask_dir):
    """
    Visualize bounding boxes extracted from masks over the given image.
    
    Args:
        img_path (str): Path to the image file.
        mask_dir (str): Path to the directory containing corresponding masks.
    """
    # Load the image
    image = Image.open(img_path).convert("RGB")
    img_id = os.path.basename(img_path).split('.')[0]
    
    # Load masks
    mask_files = glob.glob(os.path.join(mask_dir, f"{img_id}/*.png"))
    
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    
    # Loop over each mask
    for mask_path in mask_files:
        # Load mask
        mask = Image.open(mask_path)
        mask_np = np.array(mask)
        
        # Use only the first channel if multiple channels exist
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        
        # Find the bounding box
        coords = np.argwhere(mask_np > 0)  # Non-zero pixel locations
        if coords.shape[0] == 0:
            print(f"Skipping empty mask: {mask_path}")
            continue
        
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        y_min, x_min = bbox_min.tolist()
        y_max, x_max = bbox_max.tolist()
        
        # Add the bounding box as a rectangle
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add a label for the bounding box
        obj_name = os.path.basename(mask_path).split('_')[-1].split('.')[0]
        ax.text(x_min, y_min - 10, obj_name, color='red', fontsize=12, backgroundcolor="white")
    
    # Display the image with bounding boxes
    plt.axis('off')
    plt.savefig('boxes.png')


# Example Usage
if __name__ == "__main__":
    # Path to the image
    img_path = "data/coco_gsam_img/train/000000116521.jpg" # 000000225713.jpg000000228867.jpg  000000290959.jpg  000000350017.jpg  000000405225.jpg  000000462750.jpg  000000522704.jpg  000000580651.jpg
    ''' 000000056850.jpg  000000116049.jpg  000000170706.jpg  000000228994.jpg  000000290967.jpg  000000350133.jpg  000000405291.jpg  000000463189.jpg  000000522741.jpg  000000580706.jpg
000000057016.jpg  000000116126.jpg  000000170765.jpg  000000229035.jpg  000000291028.jpg  000000350134.jpg  000000405314.jpg  000000463414.jpg  000000522868.jpg  000000580822.jpg
000000057091.jpg  000000116353.jpg  000000171012.jpg  000000229084.jpg  000000291048.jpg  000000350160.jpg  000000405541.jpg  000000463669.jpg  000000522909.jpg  000000580843.jpg
000000057100.jpg  000000116521.jpg  000000171026.jpg  000000229286.jpg  000000291207.jpg  000000350245.jpg  000000405630.jpg  000000463670.jpg  000000522935.jpg  000000580971.jpg
000000057283.jpg  000000116675.jpg  000000171058.jpg  000000229362.jpg  000000291752.jpg  000000350368.jpg  000000405864.jpg  000000463898.jpg  000000523114.jpg  000000581153.jpg
000000057308.jpg  000000116696.jpg  000000171107.jpg  000000229494.jpg  000000291822.jpg  000000350481.jpg  000000406026.jpg  000000464006.jpg  000000523164.jpg  000000581422.jpg
000000057381.jpg  000000117094.jpg  000000171297.jpg  000000229633.jpg  000000292102.jpg  000000350522.jpg  000000406121.jpg  000000464526.jpg  000000523174.jpg  000000581451.jpg
000000057445.jpg  000000117108.jpg  000000171309.jpg  000000229707.jpg  000000292125.jpg  000000350592.jpg  000000406315.jpg  000000464558.jpg  000000523201.jpg  metadata.jsonl
000000057550.jpg  000000117137.jpg  000000171330.jpg  000000229740.jpg  000000292242.jpg  000000350727.jpg  000000406676.jpg  000000464871.jpg  000000523225.jpg
000000057639.jpg  000000117302.jpg  000000171483.jpg  '''
    
    # Path to the corresponding mask directory
    mask_dir = "data/coco_gsam_seg"
    
    visualize_bounding_boxes(img_path, mask_dir)
