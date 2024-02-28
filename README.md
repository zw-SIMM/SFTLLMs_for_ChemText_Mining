# SFTLLMs_for_ChemText_Mining

## Download
```bash
git clone https://github.com/zw-SIMM/SFTLLMs_for_chemtext_mining
cd SFTChatGPT_for_chemtext_mining
```

## Datasets

Preprocessed data have been placed in corresponding folders:

- ```Paragraph2Comound/```

- ```Paragraph2RXNRole/prod/``` and ```Paragraph2RXNRole/role/```

- ```Paragraph2MOFInfo/```

- ```Paragraph2NMR/```

- ```Paragraph2Action/``` dataset is derived from pistachio dataset, which is available upon request.

## Finetuning-ChatGPT

### Environment (OS: Windows or Linux)

```bash
pip install openai
pip install pandas
```
Note: The fine-tuning code has been slightly different as the version of openai updated to v1.0.0+.

Here, we provide the latest code.

### Implementation

Specific scripts  for each task are in the corresponding folders.
All notebooks of fine-tuning and prompt engineering LLMs (GPT-4, GPT-3.5) as well as evaluating for each task will be released soon...

####  Demo of Finetuning-ChatGPT on small data

Here, we gave an example notebook of fine-tuning ChatGPT on 25 Paragraph2NMR data in ```demo/fine-tuning_chatgpt_on_25_paragraph2NMR_data.ipynb```, including:

 - Preprocessing
 - Training
 - Inferencing
 - Evaluating

## Language Model Baselines

### Environment (Linux)

Finetuning codes including ChatGPT (GPT-3.5-Turbo), llama2, bart, and t5 will be released soon ...

