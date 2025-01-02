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
    plt.savefig('boxes_test.png')


# Example Usage
if __name__ == "__main__":
    print("debut")
    # Path to the image
    img_path = "data/coco_gsam_img/train/000000116675.jpg"
    
    # Path to the corresponding mask directory
    mask_dir = "data/coco_gsam_seg"
    
    visualize_bounding_boxes(img_path, mask_dir)
