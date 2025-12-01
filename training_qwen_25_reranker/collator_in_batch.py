import random
import torch
from torchvision import transforms
import torchvision
from PIL import Image

class RerankerCollatorVLM:
    def __init__(self, processor, max_length=128, num_negatives=4):
        self.processor = processor
        self.max_length = max_length
        self.num_negatives = num_negatives
    
    def __call__(self, batch):
        queries = []
        images = []
        labels = []
        prepared_image_list = [torchvision.transforms.functional.to_pil_image(item["image"]) for item in batch]
      
        # Mine negatives
        for i, item in enumerate(batch):
            query = "<|im_start|>system\nYou will be given an picture and a query. Answer 'Yes' if the answer to the query can be found in the picture, else 'No'<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Query : "+item["query"]+" \nAre the picture and query related ?<|im_end|>\n<|im_start|>assistant\n"
            pos_image = prepared_image_list[i]

            queries.append(query)
            images.append(pos_image)
            labels.append(1)

            negative_indices = random.sample([j for j in range(len(batch)) if j != i], self.num_negatives)
            for neg_idx in negative_indices:
                queries.append(query)
                images.append(prepared_image_list[neg_idx])
                labels.append(0)

        inputs = processor(
                    text=queries,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )

        inputs["labels"] = torch.tensor(labels, dtype=torch.float)

        matches = (inputs["input_ids"] == 151645)
        inputs["target_token_positions"] = matches.float().argmax(dim=1)

        batch_size = inputs["input_ids"].shape[0]

        # Artificial pixel_values batching
        tensor = inputs["pixel_values"]
        original_length = tensor.shape[0]
        d, embedd_size = tensor.shape
        num_batches = (d + batch_size - 1) // batch_size
        batches = list(torch.split(tensor, batch_size))
        
        if batches[-1].shape[0] < batch_size:
            padding_needed = batch_size - batches[-1].shape[0]
            padding = torch.zeros((padding_needed, embedd_size), dtype=tensor.dtype, device=tensor.device)
            batches[-1] = torch.cat([batches[-1], padding], dim=0)
            
        stacked_tensor = torch.stack(batches)
        batched_tensor = stacked_tensor.transpose(0, 1)
        inputs["pixel_values"] = batched_tensor
        inputs["original_length"] = torch.full((batch_size,), original_length, dtype=torch.int)

        return inputs
