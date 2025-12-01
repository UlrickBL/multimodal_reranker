from torch import nn

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

        yes_logits = selected_logits[:, token_id_yes]
        no_logits  = selected_logits[:, token_id_no]
        
        logit_diff = yes_logits - no_logits

        prob_yes = torch.sigmoid(logit_diff)
    
        return prob_yes
