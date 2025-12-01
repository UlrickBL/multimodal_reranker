import random
import torch
from PIL import Image
from torchvision.transforms import ToPILImage 

to_pil = ToPILImage()


def resize_if_needed(img: Image.Image, max_side: int = 800) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    new_size = (round(w * scale), round(h * scale))
    return img.resize(new_size, Image.LANCZOS)
    

class RerankerCollatorVLM:
    def __init__(self, processor,corpus_dict, max_length=128, num_negatives=4):
        self.processor = processor
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.corpus_dict = corpus_dict

    def _get_pos_image_from_item(self, item: dict) -> Image.Image:
        if 'image' in item:
            return item['image']
        if 'pos_id' in item and self.corpus_dict:
            return self.corpus_dict[int(item['pos_id'])]
        return None
        
    def __call__(self, batch):
        queries = []
        images = []
        labels = []

        batch_pos_images = [self._get_pos_image_from_item(item) for item in batch]
        unique_pos_images = []
        for img in batch_pos_images:
            if img is not None:
                if not any(img is unique_img for unique_img in unique_pos_images):
                    if isinstance(img, torch.Tensor):
                        img_obj = to_pil(img) 
                    else:
                        img_obj = img
                    unique_pos_images.append(img_obj)

        for i, item in enumerate(batch):
            query_prompt = (
                "<|im_start|>system\nYou will be given an picture and a query. "
                "Answer 'Yes' if the answer to the query can be found in the picture, else 'No'<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"Query : {item['query']} \nAre the picture and query related ?<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            
            if 'pos_id' in item and self.corpus_dict:
                pos_img = resize_if_needed(self.corpus_dict[int(item["pos_id"])])
                queries.append(query_prompt)
                images.append(pos_img)
                labels.append(1.0)

                neg_id = int(item["neg_ids"][0])
                neg_img = resize_if_needed(self.corpus_dict[neg_id])
                queries.append(query_prompt)
                images.append(neg_img)
                labels.append(0.0)

            elif 'image' in item:
                pos_img_obj = to_pil(item["image"]).convert('RGB')
                pos_img_resized = resize_if_needed(pos_img_obj)
                queries.append(query_prompt)
                images.append(pos_img_resized)
                labels.append(1.0)

                #Mine
                neg_img = None
                
                if len(unique_pos_images) > 1:
                    candidate_negs = [img for img in unique_pos_images if img is not pos_img_obj]
                    if candidate_negs:
                        neg_img = resize_if_needed(random.choice(candidate_negs))
                
                if neg_img is None and self.corpus_dict:
                    corpus_ids = list(self.corpus_dict.keys())
                    random_corpus_id = random.choice(corpus_ids)
                    neg_img = resize_if_needed(self.corpus_dict[random_corpus_id])

                queries.append(query_prompt)
                images.append(neg_img)
                labels.append(0.0)
            
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