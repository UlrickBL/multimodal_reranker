import torch
from PIL import Image
from torch import nn
import torch
from PIL import Image
import torch
from PIL import Image
from transformers.image_utils import load_image
import time

def load_images(images, lazy_load: bool = True):
   # Disable PIL DecompositionBomb threshold for reading large images.
   pil_max_px = Image.MAX_IMAGE_PIXELS
   Image.MAX_IMAGE_PIXELS = None

   images_batch = []
   for image in images:
       if isinstance(image, Image.Image):
           images_batch.append(image)
       else:
           pil_image = load_image(image)
           if lazy_load:
               images_batch.append(pil_image)
           else:
               # avoid Too many open files error
               images_batch.append(pil_image.copy())
               pil_image.close()
   Image.MAX_IMAGE_PIXELS = pil_max_px

   return images_batch
   
class Qwen3Reranker(nn.Module):
   def __init__(self, base_model, processor, yes_token_id=9454, no_token_id=2753):
       super().__init__()
       self.base_model = base_model
       self._processor = processor
       self.yes_token_id = yes_token_id
       self.no_token_id = no_token_id

       original_head = base_model.lm_head
       W = original_head.weight

       w_yes = W[yes_token_id]
       w_no  = W[no_token_id]
       
       hidden_dim = w_yes.shape[0]
   
       has_bias = original_head.bias is not None

       self.score_head = nn.Linear(hidden_dim, 2, bias=has_bias)
       
       with torch.no_grad():
           self.score_head.weight[0] = w_yes
           self.score_head.weight[1] = w_no
           if has_bias:
               self.score_head.bias[0] = original_head.bias[yes_token_id]
               self.score_head.bias[1] = original_head.bias[no_token_id]

       dtype = next(base_model.parameters()).dtype
       device = next(base_model.parameters()).device
       self.score_head.to(device=device, dtype=dtype)
       self.base_model.lm_head = nn.Identity()

   def forward(self, input_ids, pixel_values, attention_mask, image_grid_thw, original_length=None, labels=None):
       if len(pixel_values.shape) == 3:
           pixel_values = pixel_values.transpose(0, 1).reshape(-1, pixel_values.shape[-1])
           pixel_values = pixel_values[:original_length[0].item()]

       # Because lm_head is Identity, outputs.logits contains HIDDEN STATES
       outputs = self.base_model(
           input_ids=input_ids,
           pixel_values=pixel_values,
           image_grid_thw=image_grid_thw,
           attention_mask=attention_mask,
           use_cache=False
       )

       hidden_states = outputs.logits 

       batch_size = hidden_states.size(0)
       lengths = attention_mask.sum(dim=1)
       token_pos = lengths - 1
       batch_indices = torch.arange(batch_size, device=hidden_states.device)

       last_token_hidden = hidden_states[batch_indices, token_pos] # [B, Hidden]
       logits = self.score_head(last_token_hidden) # [B, 2] (Row 0=Yes, Row 1=No)

       yes_logits = logits[:, 0]
       no_logits  = logits[:, 1]
       
       prob_yes = torch.sigmoid(yes_logits - no_logits)

       return prob_yes

   @torch.no_grad()
   def compute_score(
       self,
       pairs,
       batch_size = 8,
       show_progress = False,
   ):

       if isinstance(pairs[0], str):
           pairs = [pairs]

       all_scores = []

       device = next(self.parameters()).device

       batch_iter = range(0, len(pairs), batch_size)
       if show_progress:
           from tqdm import trange

           batch_iter = trange(0, len(pairs), batch_size, desc="Computing scores")

       for start_index in batch_iter:
           
           mini_batch = pairs[start_index : start_index + batch_size]

           batch_inputs = []
           for question, d in mini_batch:
               queries_prepared = "<|im_start|>system\nYou will be given an picture and a query. Answer 'Yes' if the answer to the query can be found in the picture, else 'No'<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Query : "+question+" \nAre the picture and query related ?<|im_end|>\n<|im_start|>assistant\n"
               batch_inputs.append(queries_prepared)

           batch_images = load_images([d for (q, d) in mini_batch])
           
           batch = self._processor(
               text=batch_inputs,
               images=batch_images,
               return_tensors="pt",
               padding=True,
               truncation=True,
           )

           # move the batch to the correct device
           batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

           scores = self.forward(**batch).view(-1).cpu().float().numpy()
           
           all_scores.extend(scores.tolist())
       if len(all_scores) == 1:
           return all_scores[0]
       return all_scores
