import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import  AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
import logging
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from transformers import TrainerCallback

# Data Loading and Preprocessing
train_df = pd.read_csv("data/role/train.csv")
test_df = pd.read_csv("data/role/test.csv")
source_text = "input"
target_text = "output"
instruction = f"{source_text}2{target_text}: "
instruction = "annotate the reaction roles in the paragraph. "

train_df['text'] = f'<s>[INST] {instruction}' + train_df[source_text] + " [/INST] " + train_df[target_text] +  "!!! </s>"
test_df['text'] = f'<s>[INST] {instruction}' + test_df[source_text] + " [/INST] " + test_df[target_text] +  "!!! </s>"

train_dataset = Dataset.from_dict(train_df[['text']].astype(str))
test_dataset = Dataset.from_dict(test_df[['text']].astype(str))
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
print(dataset)

################################################################################
# Parameters Setting
################################################################################
# QLoRA parameters
lora_r = 64                         # LoRA attention dimension (8, 16, 64, larger is better)
lora_alpha = 128                    # Alpha parameter for LoRA scaling (lora_r*2)
lora_dropout = 0.1                  # Dropout probability for LoRA layers

# bitsandbytes parameters
use_4bit = True                     # Activate 4-bit precision base model loading
bnb_4bit_compute_dtype = "float16"  # Compute dtype for 4-bit base models   
bnb_4bit_quant_type = "nf4"         # Quantization type (fp4 or nf4)
use_nested_quant = False            # Activate nested quantization for 4-bit base models (double quantization)

# TrainingArguments parameters
num_train_epochs = 20
save_steps = 0                      # Save checkpoint every X updates steps
logging_steps = 25                  # Log every X updates steps

fp16 = False                        # Enable fp16/bf16 training (set fp16 to True with an V100)
bf16 = True                         # Enable fp16/bf16 training (set bf16 to True with an A100)

per_device_train_batch_size = 2     # Batch size per GPU for training
per_device_eval_batch_size = 2      # Batch size per GPU for evaluation
gradient_accumulation_steps = 1     # Number of update steps to accumulate the gradients for
gradient_checkpointing = True       # Enable gradient checkpointing

max_grad_norm = 0.3                 # Maximum gradient normal (gradient clipping)
learning_rate = 1e-5                # Initial learning rate (AdamW optimizer)
weight_decay = 0.001                # Weight decay to apply to all layers except bias/LayerNorm weights

optim = "paged_adamw_32bit"         # Optimizer to use
lr_scheduler_type = "cosine"        # Learning rate schedule

max_steps = -1                      # Number of training steps (overrides num_train_epochs)
warmup_ratio = 0.03                 # Ratio of steps for a linear warmup (from 0 to learning rate)

group_by_length = True              # Group sequences into batches with same length (Saves memory and speeds up training considerably)

# SFT parameters
max_seq_length = 4096               # Maximum sequence length to use (default 1024)
packing = False                     # Pack multiple short examples in the same input sequence to increase efficiency
device_map = {"": 0}                # Load the entire model on the GPU 0, or "auto"

# Model Version (Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2, llama-2-13b-chat-hf ...)
model_name = "/home/zhangwei/pretrained_models/llama-2-13b-chat-hf"     # Path of the pretrained model downloaded from Hugging Face
new_model_dir = f"saved_models/llama-2-13b-chat-hf_qlora/train_{len(train_df)}_lora_r{lora_r}_lr{learning_rate}"  # Fine-tuned model name
output_dir = new_model_dir                                              # Output directory where the model predictions and checkpoints will be stored

################################################################################
# Train
################################################################################
# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path = model_name,
    quantization_config = bnb_config,
    device_map = device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"   # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    logging_dir = output_dir + "/logs/",
    evaluation_strategy = "epoch",            
    save_strategy = "epoch",
    num_train_epochs = num_train_epochs,
    save_total_limit = num_train_epochs,
    per_device_train_batch_size = per_device_train_batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,
    optim = optim,
    #save_steps=save_steps,
    logging_steps = logging_steps,
    learning_rate = learning_rate,
    weight_decay = weight_decay,
    fp16 = fp16,
    bf16 = bf16,
    max_grad_norm = max_grad_norm,
    max_steps = max_steps,
    warmup_ratio = warmup_ratio,
    group_by_length = group_by_length,
    lr_scheduler_type = lr_scheduler_type,
    report_to = "tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset['train'],
    eval_dataset = dataset["test"],
    peft_config = peft_config,
    dataset_text_field ="text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = training_arguments,
    packing = packing,
)

class SaveBestModelCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.best_eval_loss = float('inf')
        self.best_model_checkpoint = None
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Check if training_loss is in the logs and print it
        if 'loss' in logs:
            training_loss = logs['loss']
            logging.info(f"Epoch: {int(state.epoch)}, Step: {state.global_step}, Current training_loss: {training_loss}")

    def on_evaluate(self, args, state, control, **kwargs):
        # Check if eval_loss is in the logs
        if 'eval_loss' in state.log_history[-1]:
            eval_loss = state.log_history[-1]['eval_loss']
            logging.info(f"Epoch: {int(state.epoch)}, Step: {state.global_step}, Current eval_loss: {eval_loss}")  # Print current eval_loss with epoch and step

            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                # Save the best model
                self.best_model_checkpoint = state.global_step
                trainer.save_model(f"{args.output_dir}/best_model")
                logging.info(f"New best model saved at step {state.global_step} with eval_loss: {eval_loss}")  # Print loss of Best Model

# Create an instance of the callback
save_best_model_callback = SaveBestModelCallback()

# Training and logging
logging.basicConfig(filename=output_dir+'/training.log', level=logging.INFO)
logging.info(f"""[Device]: cuda:{os.environ["CUDA_VISIBLE_DEVICES"]}...\n""")
logging.info(f"""[Model]: Loading {model_name}...\n""")
logging.info(f"""[Outputdir]: Loading {output_dir}...\n""")

# Add the callback to the trainer
trainer.add_callback(save_best_model_callback)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model_dir)