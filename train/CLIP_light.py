from torch import nn, optim
from transformers import CLIPModel, CLIPProcessor

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

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
                #print(f"Processing mask: {mask_path}")
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



# Fine-tuning CLIP
class CLIPFineTuner:
    def __init__(self, model_name="openai/clip-vit-large-patch14", lr=5e-5, tau=0.07, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tau = tau

        # Load pre-trained CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Define optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

    def info_nce_loss(self, image_embeddings, text_embeddings):
        """Compute the InfoNCE loss for a batch."""
        # Normalize embeddings
        image_embeddings = nn.functional.normalize(image_embeddings, dim=1)
        text_embeddings = nn.functional.normalize(text_embeddings, dim=1)

        # Similarity matrix
        logits = torch.matmul(image_embeddings, text_embeddings.T) / self.tau

        # Labels for contrastive learning (diagonal elements as positives)
        labels = torch.arange(logits.size(0)).to(self.device)

        # Cross-entropy loss applied on logits
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

    def train_step(self, batch):
        """Perform a single training step."""
        rois = batch['rois']  # List of ROI images
        positive_pairs = batch['positive_pairs']
        negative_pairs = batch['negative_pairs']

        print("Batch of ROIs:", len(rois))
        print("Positive Pairs:", len(positive_pairs))  # Show total number of positive pairs
        print("Negative Pairs:", len(negative_pairs))  # Show total number of negative pairs

        # Prepare data
        inputs = self.prepare_inputs(rois, positive_pairs, negative_pairs)
        image_inputs = inputs['image_inputs'].to(self.device)
        text_inputs = inputs['text_inputs'].to(self.device)

        # Debug: Print shapes of inputs
        print(f"Image inputs shape: {image_inputs.shape}")
        print(f"Text inputs shape: {text_inputs.shape}")

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        image_outputs = self.model.get_image_features(pixel_values=image_inputs)
        text_outputs = self.model.get_text_features(input_ids=text_inputs)

        # Debug: Print embeddings
        print(f"Image embeddings shape: {image_outputs.shape}")
        print(f"Text embeddings shape: {text_outputs.shape}")

        # Compute loss
        loss = self.info_nce_loss(image_outputs, text_outputs)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def prepare_inputs(self, rois, positive_pairs, negative_pairs):
        """Prepare inputs for the model and validate tensor dimensions."""
        images = []
        texts = []

        for roi, (roi_image, pos_text) in zip(rois, positive_pairs):
            roi_tensor = self.processor(images=roi_image, return_tensors="pt")["pixel_values"]

            # Add positive pair
            images.append(roi_tensor)
            encoded_pos_text = self.processor(text=[pos_text], return_tensors="pt", padding=True, truncation=True)["input_ids"]
            texts.append(encoded_pos_text[0])  # Extract from batch

            # Add negative pairs
            for neg_text in [neg[1] for neg in negative_pairs]:
                images.append(roi_tensor)
                encoded_neg_text = self.processor(text=[neg_text], return_tensors="pt", padding=True, truncation=True)["input_ids"]
                texts.append(encoded_neg_text[0])  # Extract from batch

        # Stack tensors
        image_inputs = torch.cat(images, dim=0)
        text_inputs = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)

        # Debug: Validate tensor dimensions
        print(f"Number of image tensors: {len(images)}, Image inputs shape: {image_inputs.shape}")
        print(f"Number of text tensors: {len(texts)}, Text inputs shape: {text_inputs.shape}")

        # Validate maximum sequence length for text inputs
        max_seq_len = text_inputs.size(1)
        if max_seq_len > 77:
            print(f"Warning: Text input exceeds maximum sequence length (77). Found: {max_seq_len}")

        # Ensure image and text inputs have matching batch sizes
        if image_inputs.size(0) != text_inputs.size(0):
            raise ValueError(
                f"Mismatch in batch sizes: image_inputs={image_inputs.size(0)}, text_inputs={text_inputs.size(0)}"
            )

        return {"image_inputs": image_inputs, "text_inputs": text_inputs}
    

    def save_model(self, output_dir="clip_finetuned_model"):
        """
        Save the fine-tuned CLIP model and tokenizer to the specified directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"Model and processor saved to {output_dir}")




    def train(self, dataloader, num_epochs=10):
        """Train the model."""
        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:

                loss = self.train_step(batch)
                total_loss += loss
                
                # Debug: Print batch loss
                print(f"Batch Loss: {loss:.4f}")

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
        self.save_model()




def custom_collate_fn(batch, max_combinations=2):
    rois = [roi for item in batch for roi in item["rois"]]
    positive_pairs = [pair for item in batch for pair in item["positive_pairs"]]
    negative_pairs = [pair for item in batch for pair in item["negative_pairs"]]

    # Fractionner les combinaisons
    rois = rois[:max_combinations]
    positive_pairs = positive_pairs[:max_combinations]
    negative_pairs = negative_pairs[:max_combinations]

    return {
        "rois": rois,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
    }


# Example usage
if __name__ == "__main__":
    # Paths to dataset
    img_dir = "data/coco_gsam_img/train"
    seg_dir = "data/coco_gsam_seg"


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("dataset")
    dataset = CocoGsamDataset(img_dir=img_dir, seg_dir=seg_dir, transform=transform)
    print("dataloader")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    # Fine-tuning
    fine_tuner = CLIPFineTuner()
    print("finetuning")
    fine_tuner.train(dataloader, num_epochs=1)
