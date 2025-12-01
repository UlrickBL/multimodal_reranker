from transformers import Trainer, TrainingArguments
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import os
from transformers import TrainerCallback
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
from model_reranker import Qwen2_5Reranker
from collator import RerankerCollatorVLM

wandb.login(key="API_KEY")

os.environ["WANDB_PROJECT"] = "colqwen reranker"

min_pixels = 256*28*28
max_pixels = 1080*28*28
dataset_size = 2000
batch_size = 2
epochs = 2
num_negatives = 1
accumulation_steps = 4
max_steps = (dataset_size // batch_size) * epochs

base_model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
target_modules = ["qkv","gate_proj","up_proj","base_layer"]

class LogLossCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        if "loss" in logs:
            wandb.log({"step_loss": logs["loss"]}, step=state.global_step)
          
def compute_loss(outputs, labels,num_items_in_batch):
    loss = F.binary_cross_entropy_with_logits(outputs, labels.bfloat16())
    return loss


if name=="__main__" :
  ### MODEL ####
  
  model_qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto", output_hidden_states=True,
  ).train()
  processor = AutoProcessor.from_pretrained(base_model_path,max_pixels=max_pixels)

  lora_config = LoraConfig(
      r=32, lora_alpha=32,target_modules=target_modules, lora_dropout=0.05, bias="none",
  )
  lora_model = get_peft_model(model_qwen, lora_config)
  model = Qwen2_5Reranker(lora_model)
  model.train()

  ### DATASET ###
  from datasets import load_dataset
  ds = load_dataset("UlrickBL/vidore-subset-train-2000", split="train",streaming=True).with_format("torch")
  collator = RerankerCollatorVLM(processor, num_negatives=num_negatives)

  ### TRAINING ###
  training_args = TrainingArguments(
      output_dir="./lora-regressor",
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      optim = "adamw_torch",
      max_steps=max_steps,
      logging_dir="./logs",
      logging_strategy="steps",
      remove_unused_columns=False,
      logging_steps = 1,
      eval_strategy="steps",
      eval_steps=100,
      save_strategy="steps",
      save_steps=100,
      load_best_model_at_end=True,
      metric_for_best_model="loss",
      warmup_steps = 100,
      learning_rate = 5e-5,
      save_total_limit = 1,
      report_to="wandb",
      log_level="debug",
      save_safetensors=False,
  )
  
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset = ds,
      eval_dataset = ds_eval,
      compute_loss_func=compute_loss,
      data_collator = collator,
      callbacks=[LogLossCallback()],
  )
  
  trainer.train()
