## 1. Data for Paragraph2MOFInfo

- Data for fine-tuning LLMs are in ```data/data_for_llms```.

- Data for fine-tuning bart or T5 are in ```data/data_for_bart_or_t5```. (preprocessed beacuse the 512 length limitation, split into 11 prefix tasks )

## 2. Methods for Paragraph2MOFInfo

### Prompt Engineering ChatGPT (GPT-4, GPT-3.5-Turbo)

See in ```prompt_chatgpt_for_paragraph2MOFInfo.ipynb```

### Fine-tuning ChatGPT (GPT-3.5-Turbo)

See in ```finetune_chatgpt_for_paragraph2MOFInfo.ipynb```

### Full Parameter Fine-tuning Open-source Large Language Models (Mistral, Llama3, Llama2)

Training Code in ```finetune_llms_full_for_paragraph2MOFInfo.py``

Inferencing Code in ```vllm_inference_full_finetuned_llms.ipynb```

### Parameter Efficient Fine-tuning (PEFT) Open-source Large Language Models (Mistral, Llama3, Llama2)

Training Code in ```finetune_llms_peft_for_paragraph2MOFInfo.py``

Inferencing Code in ```vllm_inference_peft_finetuned_llms.ipynb```

### Fine-tuning Language Models (T5, Bart)

See in ```finetune_bart_or_t5_for_paragraph2MOFInfo.py``

## 3. Evaluating the results of Paragraph2MOFInfo

All predictions will be saved in ```results/predictions```

Evalutating codes for LLMs (ChatGPT, Llama, Mistral) are in ```evaluate_llms_paragraph2MOFInfo_all.ipynb```

Evalutating codes for LM (Bart or T5) are in ```evaluate_bart_or_t5_Paragraph2NMR.ipynb```