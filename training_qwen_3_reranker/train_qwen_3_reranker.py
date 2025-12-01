from transformers import Trainer, TrainingArguments
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
import wandb
import os
from transformers import TrainerCallback
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from multimodal_reranker.model.qwen_3_reranker import Qwen3Reranker
from multimodal_reranker.training.collator_in_batch import RerankerCollatorVLM
from datasets import load_dataset

wandb.login(key="API_KEY")

os.environ["WANDB_PROJECT"] = "colqwen reranker"

min_pixels = 256*28*28
max_pixels = 720*28*28
                   
batch_size = 6
epochs = 2
num_negatives = 1
accumulation_steps = 10

base_model_path = "Qwen/Qwen3-VL-2B-Instruct"
target_modules = ["qkv","gate_proj","up_proj","base_layer"]

class LogLossCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        if "loss" in logs:
            wandb.log({"step_loss": logs["loss"]}, step=state.global_step)
          
def compute_loss(outputs, labels,num_items_in_batch):
    loss = F.binary_cross_entropy_with_logits(outputs, labels.bfloat16())
    return loss / accumulation_steps

def load_image_from_item(item):
    image = item["image"]
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = image.transpose(1, 2, 0)
        return Image.fromarray(image)
    else:
        return Image.open(image)


if name=="__main__" :
    ### MODEL ####
    model_qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    ).train()
    processor = AutoProcessor.from_pretrained(base_model_path,max_pixels=max_pixels)

    lora_config = LoraConfig(
        r=16, lora_alpha=32,target_modules=target_modules, lora_dropout=0.05, bias="none",
    )
    lora_model = get_peft_model(model_qwen, lora_config)
    model = Qwen3Reranker(lora_model)
    model.train()

    ### DATASET ###
    ds_mm = load_dataset("UlrickBL/mm_reranker_rl_train", "train")["train"].with_format("torch").shuffle(seed=42)
    corpus_dataset = load_dataset("UlrickBL/mm_reranker_rl_train", "corpus")["corpus"].with_format("torch")
    corpus_dict = {i: load_image_from_item(item) for i, item in enumerate(corpus_dataset)}
    ds_vidore = load_dataset("UlrickBL/vidore-subset-train-2000", "default")["train"].with_format("torch").shuffle(seed=42)
    ds_eval = load_dataset("vidore/colpali_train_set", split="test",streaming=True).with_format("torch")
    
    train_ds_combined = ConcatDataset([ds_mm, ds_vidore])
    collator = RerankerCollatorVLM(processor, corpus_dict = corpus_dict, num_negatives=num_negatives)
    max_steps = (dataset_size // (batch_size * accumulation_steps)) * epochs

    ### TRAINING ###
    training_args = TrainingArguments(
        output_dir="./lora-regressor",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        optim = "adamw_torch",
        max_steps=max_steps,
        logging_dir="./logs",
        gradient_accumulation_steps=accumulation_steps,
        #evaluation_strategy="no",
        logging_strategy="steps",
        max_grad_norm = 0.1,
        remove_unused_columns=False,
        logging_steps = 1,
        weight_decay  = 1e-2, 
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        warmup_steps = 20,
        learning_rate = 1e-5,
        save_total_limit = 1,
        report_to="wandb",
        log_level="debug",
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset = train_ds_combined,
        eval_dataset = ds_eval,
        compute_loss_func=compute_loss,
        data_collator = collator,
        callbacks=[LogLossCallback()],
    )

    trainer.train()
