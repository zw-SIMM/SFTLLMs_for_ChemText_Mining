# SFTChatGPT_for_chemtext_mining
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

### Implementation

Specific scripts  for each task are in the corresponding folders.

####  Demo of Finetuning-ChatGPT on small data

Here, we gave an example notebook of fine-tuning ChatGPT on 25 Paragraph2NMR data in ```demo/SFTChatGPT_for_ChemText_Mining.ipynb```, including:

 - Preprocessing
 - Training
 - Inferencing
 - evaluating

## Language Model Baselines

### Environment (Linux)

Codes will be released soon ...

