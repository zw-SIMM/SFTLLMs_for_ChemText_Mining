## 1. Data for Paragraph2Prod

All Data are in ```data/prod```.

## 2. Methods for Paragraph2Comound

### Prompt Engineering ChatGPT (GPT-4, GPT-3.5-Turbo)

See in ```prompt_chatgpt_for_paragraph2prod.ipynb```

### Fine-tuning ChatGPT (GPT-3.5-Turbo)

See in ```finetune_chatgpt_for_paragraph2prod.ipynb```

### Full Parameter Fine-tuning Open-source Large Language Models (Mistral, Llama3, Llama2)

Training Code in ```finetune_llms_full_for_paragraph2prod.py``

Inferencing Code in ```vllm_inference_full_finetuned_llms.ipynb```

### Parameter Efficient Fine-tuning (PEFT) Open-source Large Language Models (Mistral, Llama3, Llama2)

Training Code in ```finetune_llms_peft_for_paragraph2prod.py``

Inferencing Code in ```vllm_inference_peft_finetuned_llms.ipynb```

### Fine-tuning Language Models (T5, Bart)

See in ```finetune_bart_or_t5_for_paragraph2prod.py``

## 3. Evaluating the results of Paragraph2Comound

All predictions will be saved in ```results/predictions```

Evaluating codes are in ```evaluate_Paragraph2Prod.ipynb```