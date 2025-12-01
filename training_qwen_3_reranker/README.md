---
pipeline_tag: text-classification
tags:
- vidore
- reranker
- qwen3_vl
language:
- multilingual
base_model:
- Qwen/Qwen3-VL-2B-Instruct
inference: false
library_name: transformers
license: apache-2.0
---

# Qwen3RerankerMM-2B
* Model Name: Qwen3RerankerMM-2B
* Model Type: Multilingual Multimodal Reranker
* Base Model: Qwen/Qwen3-VL-2B-Instruct
* Architecture Modifications: LoRA fine-tuned classifier on the "yes" vs "no" token logits, using sigmoid for scoring (inspired by Qwen text reranker : https://qwenlm.github.io/blog/qwen3-embedding)
* Training Setup: Resource-constrained (single A100 80GB, batch size 6)
* Parameters : by merging lora and slicing LM heads : total number of parameters is 1.8B instead of 2.1B of the base model

## Model Description
Qwen3RerankerMM is a multilingual reranking model trained with a simple but effective strategy inspired by Alibaba Qwen Text Reranker. Instead of adding a classification head, it computes relevance scores using a sigmoid function on the logit difference between the tokens “yes” and “no.”

This model is designed to be lightweight, general-purpose, and compatible with multimodal QwenVL.

## Training Details
* Training Dataset: UlrickBL/mm_reranker_rl_train - from ibm-research/REAL-MM-RAG_FinTabTrainSet_rephrased with hard negative mining (1,000 randomly sampled training examples, hard negative mining with ColQwen2.5) and UlrickBL/vidore-subset-train-2000 (2,000 randomly sampled training examples)

* Epochs: 2

* Batch Size: 6

* Gradient accumulation : 2

* Negative Mining: Mix of In-batch hard negative (Vidore) and ColQwen mined hard negatives (RealMM)

* Loss Function: Binary classification (logit diff between “yes” and “no” passed through sigmoid)

* Optimizer: AdamW

* Fine-Tuning Method: LoRA (r 16 - alpha 32) + transformers trainer (with specific trick to deal with Qwen 3 pixel_values being unbatched)

  

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/kHU-Rtd98W_ApLmdfjDkY.png)

* Hardware: Single A100 GPU (80GB)

## Evaluation Results (NDCG@5)
| Dataset                  | Jina Reranker m0 | Qwen3RerankerMM-2B |
|--------------------------|------------------|---------------------|
| UlrickBL/vidore_benchmark_2_esg_reports_human_labeled_v2_reranker_adapted      | **0.851**            | 0.804               |
| UlrickBL/REAL-MM-RAG_FinSlides_BEIR_reranker_adapted (rephrased level 1)     | 0.873            | **0.906**               |
| UlrickBL/vidore_benchmark_economics_reports_v2_reranker_adapted   | 0.735            | **0.813**               |
| UlrickBL/vidore_benchmark_arxivqa_reranker_adapted                 | 0.767            | **0.778**               |

Note: Despite smaller training data, diversity of data and compute, Qwen3RerankerMM-2B shows overall superior performance (even in hard rephrased setup).

## Inference comparison
For 50 examples with each being a batch of 25 pairs image / query with flash attention on a A100 40GB SXM4 :

* Jina Reranker m0 : 2 minutes 12 seconds
* Qwen3RerankerMM-2B : 1 minute 41 seconds

## Load model for efficient inference (merge lora + with full sliced LM heads - same performance, really quick, removes 300M parameters)

 ```python
import torch
from PIL import Image
from transformers import BitsAndBytesConfig
from torch import nn
from peft import PeftModel, PeftConfig
from huggingface_hub import hf_hub_download
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
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

# Load the model
min_pixels = 256*28*28
max_pixels = 720*28*28
model_qwen = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2", device_map="cuda"
).eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct",max_pixels=max_pixels)

base = PeftModel.from_pretrained(model_qwen, "UlrickBL/Qwen3RerankerMM-2B")

base = base.merge_and_unload()

model = Qwen3Reranker(base_model=base,processor=processor)

model=model.to("cuda")

  ```


## Load model (with full LM heads)

 ```python
    import torch
    from PIL import Image
    from torch import nn
    from peft import PeftModel, PeftConfig
    from huggingface_hub import hf_hub_download
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info

    
    class Qwen3Reranker(nn.Module):
      def __init__(self, base_model,processor):
          super().__init__()
          self.base_model = base_model
          self._processor = processor
  
      def forward(self, input_ids,pixel_values, attention_mask,image_grid_thw,original_length=None,labels=None):
          # Readapt pixel values, mostly for batching in training collator
          if len(pixel_values.shape)==3 :
              pixel_values = pixel_values.transpose(0, 1).reshape(-1, pixel_values.shape[-1])
              pixel_values = pixel_values[:original_length[0].item()]
  
              
          generated_ids = self.base_model.forward(input_ids=input_ids,pixel_values=pixel_values,image_grid_thw=image_grid_thw, attention_mask=attention_mask)
  
          logits =generated_ids.logits
          batch_size = logits.size(0)
          batch_indices = torch.arange(batch_size, device=logits.device)
          
          lengths = attention_mask.sum(dim=1)
          token_pos = lengths -1
          token_id_yes = 9454
          token_id_no = 2753
          
          selected_logits = logits[batch_indices, token_pos]
          
          yes_logits = selected_logits[:, token_id_yes]  # shape: [batch_size]
          no_logits  = selected_logits[:, token_id_no]   # shape: [batch_size]
  
          logit_diff = yes_logits - no_logits
  
          prob_yes = torch.sigmoid(logit_diff)
      
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
  
              batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
  
              scores = self.forward(**batch).view(-1).cpu().float().numpy()
              
              all_scores.extend(scores.tolist())
          if len(all_scores) == 1:
              return all_scores[0]
          return all_scores


    
    # Load the model
    min_pixels = 256*28*28
    max_pixels = 720*28*28
    model_qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2", device_map="cuda"
    ).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct",max_pixels=max_pixels)
    
    base = PeftModel.from_pretrained(model_qwen, "UlrickBL/Qwen3RerankerMM-2B")

    model_full = Qwen3Reranker(base_model=base,processor=processor)

    model=model.to("cuda")
    model.eval()
 ```

## Inference code
  ```python
    query = "slm markdown"
    documents = [
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png",
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png",
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/wired-preview.png",
        "https://jina.ai/blog-banner/using-deepseek-r1-reasoning-model-in-deepsearch.webp"
    ]
    
    # construct sentence pairs
    image_pairs = [[query, doc] for doc in documents]

    scores = model.compute_score(image_pairs)
  ```


