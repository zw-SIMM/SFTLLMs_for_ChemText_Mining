# Finetuning Bart or T5
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import pandas as pd 
import numpy as np
import torch
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
from datasets import load_metric, Dataset, DatasetDict
import warnings
warnings.filterwarnings('ignore')

# Configuration
class CFG:
    # Model Configuration
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_train_epochs = 50
    save_total_limit = 50
    batch_size = 8
    learning_rate = 1e-5
    max_input_length = 512
    max_target_length = 512
    weight_decay = 0.01
    save_strategy = "epoch"
    evaluation_strategy = "epoch"
    interval_eval_epoch = 1                               # the number of interval epochs to evaluate (inference)
    model_name = "t5-base"                                # "bart-base" or "t5-base"
    pretrained_dir = "/home/zhangwei/pretrained_models/"  # Path of the pretrained model downloaded from Hugging Face
    saved_models_dir = f"saved_models/{model_name}/train_200_{model_name}_lr_{learning_rate}_epoch_{num_train_epochs}_bs_{batch_size}_intervel_{interval_eval_epoch}/"
    output_dir = f"results/predictions/{saved_models_dir}"

    # Data Configuration
    train_file = "data/data_for_bart_or_t5/t5_bart_train_200_one_column_lstrip_add_space.csv"
    test_file = "data/data_for_bart_or_t5/t5_bart_test_300_one_column_lstrip_add_space.csv"
    source_text_column = "Paragraph"
    target_text_column = "output"

def load_data():
    train_df = pd.read_csv(CFG.train_file)
    test_df = pd.read_csv(CFG.test_file)
    train_dataset = Dataset.from_dict(train_df.astype(str))
    test_dataset = Dataset.from_dict(test_df.astype(str))
    datasets = DatasetDict({"train": train_dataset, "test": test_dataset})
    print(datasets)
    return datasets

def tokenize_and_encode(tokenizer, datasets):
    def tokenize_function(examples):
        model_inputs = tokenizer(examples[CFG.source_text_column], max_length=CFG.max_input_length, truncation=True)
        model_labels = tokenizer(examples[CFG.target_text_column], max_length=CFG.max_target_length, truncation=True)
        model_inputs["labels"] = model_labels["input_ids"]
        return model_inputs
    return datasets.map(tokenize_function, batched=True)

def logging_config():
    logging.info("Configuration Details:")
    for attr in dir(CFG):
        # Filter out private attributes and methods
        if not attr.startswith("__") and not callable(getattr(CFG, attr)):
            logging.info(f"{attr}: {getattr(CFG, attr)}")

# Custom Callback
class CustomCallback(TrainerCallback):    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            training_loss = logs['loss']
            logging.info(f"Epoch: {int(state.epoch)}, Step: {state.global_step}, Current training_loss: {training_loss}")

        if 'eval_loss' in state.log_history[-1]:
            eval_loss = state.log_history[-1]['eval_loss']
            logging.info(f"Epoch: {int(state.epoch)}, Step: {state.global_step}, Current eval_loss: {eval_loss}")

    def on_epoch_end(self, args, state, control, **kwargs):
            logging.info("Saving inference results for test_set...")
            output = self._trainer.predict(self._trainer.eval_dataset)
            epoch = int(state.epoch)

            if epoch % CFG.interval_eval_epoch == 0 :
                # Decode generated summaries into text
                decoded_ids = output.predictions

                # Replace -100 in the labels as we can't decode them
                decoded_ids = np.where(decoded_ids != -100, decoded_ids, tokenizer.pad_token_id)
                decoded_texts = tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)
                paragraphs = [i[CFG.source_text_column] for i in self._trainer.eval_dataset]
                ground_truth = [i[CFG.target_text_column] for i in self._trainer.eval_dataset]
                prediction = [decoded_text for decoded_text in decoded_texts]

                # Save predictions to csv
                predicted_df = pd.DataFrame()
                predicted_df['Paragraph'] = paragraphs
                predicted_df['Generated Text'] = prediction
                predicted_df['Actual Text'] = ground_truth
                predicted_df.to_csv(f"{CFG.output_dir}/epoch_{epoch}.csv", index = None)

def main():
    # mkdir needed folders
    if not os.path.exists(CFG.saved_models_dir):
        os.makedirs(CFG.saved_models_dir)
    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)

    # Setup logging
    logging.basicConfig(filename = CFG.saved_models_dir+'/training.log', level = logging.INFO)
    logging_config()

    # Loading Tokenizer and Model
    print("Loading Tokenizer and Model ...")
    logging.info(f"[Device]: {CFG.device}...")
    logging.info(f"[Model]: Loading {CFG.model_name}...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{CFG.pretrained_dir}/{CFG.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(f"{CFG.pretrained_dir}/{CFG.model_name}").to(CFG.device)

    # Loading Data
    print("Loading Data ...")
    datasets = load_data()

    # Preparing Data
    print("Preparing Data ...")
    logging.info(f"[Dataset]:\n{datasets}")
    tokenized_datasets = tokenize_and_encode(tokenizer, datasets)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model = model)

    # Training Arguments
    args = Seq2SeqTrainingArguments(
        output_dir = CFG.saved_models_dir,
        logging_dir = CFG.saved_models_dir + "logs/",
        evaluation_strategy = CFG.evaluation_strategy,            
        learning_rate = CFG.learning_rate,
        per_device_train_batch_size = CFG.batch_size,
        per_device_eval_batch_size = CFG.batch_size,
        weight_decay = CFG.weight_decay,
        generation_max_length = CFG.max_target_length,
        save_strategy =  CFG.save_strategy,
        num_train_epochs = CFG.num_train_epochs,
        save_total_limit = CFG.save_total_limit,
        predict_with_generate = True,
        logging_steps = len(tokenized_datasets["train"]) // CFG.batch_size,
        push_to_hub = False,
        report_to = "tensorboard")

    # Trainer                
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset = tokenized_datasets["train"],
        eval_dataset = tokenized_datasets["test"],
        data_collator = data_collator,
        tokenizer = tokenizer)

    # Training and logging
    print("Training ...")
    trainer.add_callback(CustomCallback(trainer)) 
    trainer.train()

if __name__ == "__main__":
    main()