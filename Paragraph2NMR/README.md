## 1. Data for Paragraph2NMR

- Data for fine-tuning LLMs are in ```data/data_for_llms```.

- Data for fine-tuning bart or T5 are in ```data/data_for_bart_or_t5```. (preprocessed beacuse the 512 length limitation, split into 7 prefix tasks )

## 2. Methods for Paragraph2NMR

### Prompt Engineering ChatGPT (GPT-4, GPT-3.5-Turbo)

See in ```prompt_chatgpt_for_paragraph2NMR.ipynb```

### Fine-tuning ChatGPT (GPT-3.5-Turbo)

See in ```finetune_chatgpt_for_paragraph2NMR.ipynb```

### Full Parameter Fine-tuning Open-source Large Language Models (Llama3, Llama2, Mistral)

Training Code in ```finetune_llms_full_for_paragraph2NMR.py```

Inferencing Code in ```vllm_inference_full_finetuned_llms.ipynb```

### Parameter Efficient Fine-tuning (PEFT) Open-source Large Language Models (Llama3, Llama2, Mistral)

Training Code in ```finetune_llms_peft_for_paragraph2NMR.py```

Inferencing Code in ```vllm_inference_peft_finetuned_llms.ipynb```

### Fine-tuning Language Models (T5, Bart)

See in ```finetune_bart_or_t5_for_paragraph2NMR.py```

## 3. Evaluating the results of Paragraph2NMR

All predictions will be saved in ```results/predictions```

Evalutating codes for LLMs (ChatGPT, Llama, Mistral) are in ```evaluate_llms_Paragraph2NMR.ipynb```

Evalutating codes for LM (Bart or T5) are in ```evaluate_bart_or_t5_Paragraph2NMR.ipynb```
