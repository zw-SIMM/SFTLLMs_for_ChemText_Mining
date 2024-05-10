## 1. Data for Paragraph2Action
The ```data/few_example_data``` of Paragraph2Action dataset is a subset of https://github.com/rxn4chemistry/paragraph2actions/tree/main/test_data

The whole paragraph2action dataset ```data/hand_annotated``` is available upon request (with pistachio license).

The processed dataset is in ```data/processed_data```.

## 2. Methods for Paragraph2Action

### Prompt Engineering ChatGPT (GPT-4, GPT-3.5-Turbo)

See in ```prompt_chatgpt_for_paragraph2action.ipynb```

### Fine-tuning ChatGPT (GPT-3.5-Turbo)

See in ```finetune_chatgpt_for_paragraph2action.ipynb```

### Full Parameter Fine-tuning Open-source Large Language Models (Mistral, Llama3, Llama2)

Training Code in 

```bash
python finetune_llms_full_for_paragraph2action.py
```

Inferencing Code in ```vllm_inference_full_finetuned_llms.ipynb```

### Parameter Efficient Fine-tuning (PEFT) Open-source Large Language Models (Mistral, Llama3, Llama2)

Training Code

```bash
python finetune_llms_peft_for_paragraph2action.py
```

Inferencing Code in ```vllm_inference_peft_finetuned_llms.ipynb```

### Fine-tuning Language Models (T5, Bart)

Training Code

```bash
python finetune_bart_or_t5_for_paragraph2action.py
```


## 3. Evaluating the results of Paragraph2Action

All predictions will be saved in ```results/predictions```

Evaluating codes are in ```evaluate_for_paragraph2action.ipynb```
