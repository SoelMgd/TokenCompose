import torch
import json
import argparse
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from accelerate import PartialState

parser = argparse.ArgumentParser()

parser.add_argument("--text_file_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--img_per_prompt", type=int, default=10)

args = parser.parse_args()

model_name = args.model_name
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)

################################ Remplacement ##########################
from transformers import CLIPTextModel, CLIPTokenizer

fine_tuned_clip_model_dir = "../train/clip_finetuned_model"  # Chemin où vos poids sont sauvegardés
new_text_encoder = CLIPTextModel.from_pretrained(fine_tuned_clip_model_dir)
new_tokenizer = CLIPTokenizer.from_pretrained(fine_tuned_clip_model_dir)

# Remplacer les composants CLIP dans la pipeline
pipe.text_encoder = new_text_encoder
pipe.tokenizer = new_tokenizer
########################################################################

distributed_state = PartialState()

# Vérifier l'état du GPU
print(f"Accelerate device: {distributed_state.device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")


pipe.to(distributed_state.device)
pipe.set_progress_bar_config(disable=True)
print("Model loaded.")

text_file_path = args.text_file_path

with open(text_file_path, "r") as f:
    text_data = json.load(f)

# prepare the data
d = []
for index in range(len(text_data)):
    texts = [text_data[index]["text"] for _ in range(args.img_per_prompt)]
    for j in range(args.img_per_prompt):
        d.append((index, texts[j], j))
print("Data prepared.")

print("Start generating images.")
with distributed_state.split_between_processes(d) as data:
    for index, text, j in tqdm(data):
        img_id = "{}_{}".format(index, j)
        save_path = f"{args.output_dir}/{img_id}.png"
        image = pipe(prompt=[text]).images[0]
        image.save(save_path)
