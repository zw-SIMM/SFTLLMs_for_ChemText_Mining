## 1. Data for Paragraph2Comound

All Data are in ```data```.

For training data, we randomly sample 3 times from  10000 to  1000, 100, 10:
- ```/data/train/trial_1```
- ```/data/train/trial_2``` 
- ```/data/train/trial_3``` 

## 2. Methods for Paragraph2Comound

### Prompt Engineering ChatGPT (GPT-4, GPT-3.5-Turbo)

See in ```prompt_chatgpt_for_paragraph2compound.ipynb```

### Fine-tuning ChatGPT (GPT-3.5-Turbo)

See in ```finetune_chatgpt_for_paragraph2compound.ipynb```

### Full Parameter Fine-tuning Open-source Large Language Models (Mistral, Llama3, Llama2)

Training Code in ```finetune_llms_full_for_paragraph2compound.py``

Inferencing Code in ```vllm_inference_full_finetuned_llms.ipynb```

### Parameter Efficient Fine-tuning (PEFT) Open-source Large Language Models (Mistral, Llama3, Llama2)

Training Code in ```finetune_llms_peft_for_paragraph2compound.py``

Inferencing Code in ```vllm_inference_peft_finetuned_llms.ipynb```

### Fine-tuning Language Models (T5, Bart)

See in ```finetune_bart_or_t5_for_paragraph2compound.py``

## 3. Evaluating the results of Paragraph2Comound

All predictions will be saved in ```results/predictions```

Evaluating codes are in ```evaluate_for_paragraph2compound.ipynb```