import torch
from torch import nn
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

# Wrapper pour adapter les dimensions et conserver les méthodes d'origine
class CLIPTextWrapper(nn.Module):
    def __init__(self, text_encoder, target_dim):
        super().__init__()
        self.text_encoder = text_encoder
        self.projection = nn.Linear(512, target_dim)  # Adapter de 512 à 768

    def forward(self, input_ids, attention_mask=None):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        adapted_outputs = self.projection(outputs.last_hidden_state)
        return adapted_outputs

    def __getattr__(self, name):
        # Délègue tous les attributs/méthodes manquants au modèle d'origine
        return getattr(self.text_encoder, name)

# Charger la pipeline Stable Diffusion
model_id = "mlpc-lab/TokenCompose_SD14_A"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# Charger le modèle CLIP avec une taille de sortie de 512
new_clip_model_id = "openai/clip-vit-base-patch32"
new_text_encoder = CLIPTextModel.from_pretrained(new_clip_model_id)
new_tokenizer = CLIPTokenizer.from_pretrained(new_clip_model_id)

# Ajouter une projection pour rendre les dimensions compatibles
adapted_text_encoder = CLIPTextWrapper(new_text_encoder, target_dim=768)

# Remplacer les composants CLIP dans la pipeline
pipe.text_encoder = adapted_text_encoder
pipe.tokenizer = new_tokenizer

# Transférer la pipeline sur le GPU
pipe = pipe.to(device)

# Effectuer une inférence
prompt = "A cat and a wine glass"
image = pipe(prompt).images[0]

# Sauvegarder l'image
image.save("testing_image.png")
