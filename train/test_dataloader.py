from torch.utils.data import DataLoader
from torchvision import transforms
import torch


class CocoGsamDataset(Dataset):
    def __init__(self, img_dir, seg_dir, transform=None, target_size=(224, 224)):
        """
        Args:
            img_dir (str): Path to the directory containing images.
            seg_dir (str): Path to the directory containing segmentation masks.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_size (tuple): Size of the output ROI images (height, width).
        """
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.transform = transform
        self.target_size = target_size

        # List all images in the directory
        self.img_files = glob.glob(os.path.join(img_dir, "*.jpg"))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
            # Load image
            img_path = self.img_files[idx]
            img_id = os.path.basename(img_path).split('.')[0]
            image = Image.open(img_path).convert("RGB")

            # Load corresponding masks
            mask_dir = os.path.join(self.seg_dir, img_id)
            mask_files = glob.glob(os.path.join(mask_dir, "*.png"))

            # Prepare data
            rois = []
            positive_pairs = []
            negative_pairs = []
            names = [os.path.basename(mask).split('_')[-1].split('.')[0] for mask in mask_files]

            for mask_path in mask_files:
                # Load mask
                mask = Image.open(mask_path)
                mask_np = np.array(mask)

                # If the mask has multiple channels, use only the first channel
                if mask_np.ndim == 3:
                    #print("Using the first channel of the mask for segmentation.")
                    mask_np = mask_np[..., 0]  # Select the first channel

                # Debug: Print mask information
                print(f"Processing mask: {mask_path}")
                #print(f"Mask shape: {mask_np.shape}")
                #print(f"Non-zero values in mask: {np.count_nonzero(mask_np)}")

                # Get bounding box of the mask
                coords = np.argwhere(mask_np > 0)  # Ensure mask is binary or non-zero
                if coords.shape[0] == 0:
                    print("Skipping empty mask.")
                    continue  # Skip empty masks

                bbox_min = coords.min(axis=0)
                bbox_max = coords.max(axis=0)

                #print(f"Bounding box min: {bbox_min}, max: {bbox_max}")

                y_min, x_min = bbox_min.tolist()
                y_max, x_max = bbox_max.tolist()
            
                # Crop the ROI from the image using the bounding box
                roi = image.crop((x_min, y_min, x_max, y_max))

                # Resize the ROI to the target size
                roi = roi.resize(self.target_size, Image.ANTIALIAS)

                # Extract name from the mask file
                obj_name = os.path.basename(mask_path).split('_')[-1].split('.')[0]

                # Positive pair: (ROI, obj_name)
                positive_pairs.append((roi, obj_name))

                # Negative pairs: (ROI, other_names)
                other_names = [name for name in names if name != obj_name]
                for neg_name in other_names:
                    negative_pairs.append((roi, neg_name))

                rois.append(roi)

            # Apply transformations if any
            if self.transform:
                rois = [self.transform(roi) for roi in rois]

            return {
                "rois": rois,
                "positive_pairs": positive_pairs,
                "negative_pairs": negative_pairs
            }





def custom_collate_fn(batch):
    """
    Custom collate function to handle uneven batch data.
    """
    rois = [roi for item in batch for roi in item["rois"]]
    positive_pairs = [pair for item in batch for pair in item["positive_pairs"]]
    negative_pairs = [pair for item in batch for pair in item["negative_pairs"]]

    return {
        "rois": rois,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
    }

def validate_batch(batch):
    """
    Validate that all data in the batch has the same structure.
    """
    try:
        rois = batch["rois"]
        positive_pairs = batch["positive_pairs"]
        negative_pairs = batch["negative_pairs"]

        # Check if ROI tensors have consistent dimensions
        roi_shapes = [roi.shape for roi in rois]
        if len(set(roi_shapes)) > 1:
            print("Inconsistent ROI shapes:", roi_shapes)
        else:
            print("All ROI shapes are consistent.")

        # Check if all positive pairs are tuples with the expected structure
        for i, pair in enumerate(positive_pairs):
            if not (isinstance(pair, tuple) and len(pair) == 2):
                print(f"Inconsistent positive pair at index {i}: {pair}")

        # Check if all negative pairs are tuples with the expected structure
        for i, pair in enumerate(negative_pairs):
            if not (isinstance(pair, tuple) and len(pair) == 2):
                print(f"Inconsistent negative pair at index {i}: {pair}")

        print("Batch validation passed.")
    except Exception as e:
        print("Batch validation failed:", e)

if __name__ == "__main__":
    # Define paths
    img_dir = "data/coco_gsam_img/train"
    seg_dir = "data/coco_gsam_seg"

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = CocoGsamDataset(img_dir=img_dir, seg_dir=seg_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    # Iterate through the DataLoader and validate batch structure
    print("Testing DataLoader with validation...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        rois = batch["rois"]
        positive_pairs = batch["positive_pairs"]
        negative_pairs = batch["negative_pairs"]

        # Print dimensions of data
        print(f"  Number of ROIs: {len(rois)}")
        print(f"  Number of Positive Pairs: {len(positive_pairs)}")
        print(f"  Number of Negative Pairs: {len(negative_pairs)}")

        # Validate batch structure
        validate_batch(batch)

        # Break after first batch for testing
        break