from transformers import AutoTokenizer, CLIPTextModel
import torch
import random

class ClipTextEncoder:
    def __init__(self, device):
        model_id = "openai/clip-vit-base-patch32"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.text_model = (
            CLIPTextModel.from_pretrained(model_id)
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        class_text_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        self.class_embeds = {}
        for i in range(len(class_text_list)):
            class_text = class_text_list[i]
            self.class_embeds[i] = self.embed_text(f"a photo of {class_text}")
            

    def embed_text(self, caption):
        inputs = self.tokenizer(
            [caption],
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.text_model(**inputs)
        text_embeds = outputs.last_hidden_state
        return text_embeds
    
    def convert_class_idx(self, class_idx):
        if random.random() < 0.1: # randomly dropout class embed 10% of time
            return torch.zeros(1,77,512).to(self.device)
        return self.class_embeds[class_idx]
    
    def batch_embeds(self, class_idxs):
        embeds = []
        for idx in class_idxs:
            embeds.append(self.convert_class_idx(idx.item()))
        return torch.cat(embeds, dim=0)