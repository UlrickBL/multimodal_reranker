---
pipeline_tag: text-classification
tags:
- vidore
- reranker
- qwen25_vl
language:
- multilingual
base_model:
- Qwen/Qwen2.5-VL-3B-Instruct
inference: false
library_name: transformers
---

# MultimodalQwenLogitReranker-3B
* Model Name: MultimodalQwenLogitReranker-3B
* Model Type: Multilingual Multimodal Reranker
* Base Model: Qwen/Qwen2.5-VL-3B-Instruct
* Architecture Modifications: LoRA fine-tuned classifier on the "yes" vs "no" token logits, using sigmoid for scoring (inspired by Qwen text reranker : https://qwenlm.github.io/blog/qwen3-embedding)
* Training Setup: Resource-constrained (single A100, batch size 2)

## Model Description
QwenLogitReranker is a multilingual reranking model trained with a simple but effective strategy inspired by Alibaba Qwen Text Reranker. Instead of adding a classification head, it computes relevance scores using a sigmoid function on the logit difference between the tokens “yes” and “no.”

This model is designed to be lightweight, general-purpose, and compatible with multimodal QwenVL.

## Training Details
* Training Dataset: DocVQA (2,000 randomly sampled training examples)

* Epochs: 1

* Batch Size: 2

* Negative Mining: In-batch hard negative

* Loss Function: Binary classification (logit diff between “yes” and “no” passed through sigmoid)

* Optimizer: AdamW

* Fine-Tuning Method: LoRA + transformers trainer (with specific trick to deal with Qwen 2.5 pixel_values being unbatched)

* Hardware: Single A100 GPU

## Evaluation Results (NDCG@5)
| Dataset    | Jina Reranker m0 (Baseline) | QwenLogitReranker |
| ---------- | ------------------------ | ----------------- |
| UlrickBL/vidore_benchmark_economics_reports_v2_reranker_adapted  | 0.735                    | **0.799**         |
| UlrickBL/vidore_benchmark_2_biomedical_lectures_v2_reranker_adapted    | **0.763**   | 0.755             |
| UlrickBL/vidore_benchmark_2_esg_reports_human_labeled_v2_reranker_adapted  | **0.851**                | 0.820             |
| UlrickBL/vidore_benchmark_docvqa_reranker_adapted | **0.767**                    | 0.747             |
| UlrickBL/vidore_benchmark_2_esg_reports_v2_reranker_adapted     | **0.920**   | **0.910**         |
| Inference time (4898*2810 image, T4 GPU)    | 2.212 s  | **1.161 s**         |

Note: Despite smaller training data, diversity of data and compute, QwenLogitReranker shows competitive or superior performance, especially in Economics.

## Limitations
* Trained only on a small subset (2000 samples) of DocVQA

* One epoch of training — performance could likely improve with more compute/data

* Currently uses causal language model decoding to simulate classification; slower than embedding-based methods (making it bidertional like collama could improve performances but need compute)

## Load model

 ```python
    import torch
    from PIL import Image
    from torch import nn
    from peft import PeftModel, PeftConfig
    from huggingface_hub import hf_hub_download
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info

    
    class Qwen2_5Reranker(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        def forward(self, input_ids,pixel_values, attention_mask,image_grid_thw,original_length=None,labels=None):
            # Readapt pixel values
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
    
    # Load the model
    max_pixels = 1080*28*28
    model_qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16, device_map="auto", output_hidden_states=True,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",max_pixels=max_pixels)
    
    base = PeftModel.from_pretrained(model_qwen, "UlrickBL/qwen_vl_reranker_adapter_V2")
    
    model = Qwen2_5Reranker(base_model=base, hidden_dim=2048)
    
    model=model.to("cuda")
    model.eval()
 ```

## Inference code
  ```python
    import time

    start_time = time.time()
    
    url = "https://oto.hms.harvard.edu/sites/g/files/omnuum8391/files/2025-04/PowerPoint-Presentation-Graphic.jpg"
    
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    
    query = "<|im_start|>system\nYou will be given an picture and a query. Answer 'Yes' if the answer to the query can be found in the picture, else 'No'<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Query : "+"What is the Harvard study departement in the question ?"+" \nAre the picture and query related ?<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = processor(
                    text=[query],
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                )
    
    inputs.to("cuda")
    
    with torch.no_grad():
        batch_scores = model(**inputs)
    end_time = time.time()
    
    print(f"Time taken : {end_time - start_time:.4f} seconds")
  ```
## Future Work
* Expand training with additional multilingual and domain-specific datasets

* Increase batch size and epoch number

* Compare with last hidden state + classification layer
