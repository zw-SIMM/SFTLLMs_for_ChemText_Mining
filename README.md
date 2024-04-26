# SFTLLMs_for_ChemText_Mining

## Download
```bash
git clone https://github.com/zw-SIMM/SFTLLMs_for_chemtext_mining
cd SFTLLMs_for_chemtext_mining
```

## Datasets and Codes

Preprocessed data and fine-tuning codes have been placed in corresponding folders:

- ```Paragraph2Comound/```

- ```Paragraph2RXNRole/prod/``` and ```Paragraph2RXNRole/role/```

- ```Paragraph2MOFInfo/```

- ```Paragraph2NMR/```

- ```Paragraph2Action/``` dataset is derived from pistachio dataset, which is available upon request.

## Fine-tuning ChatGPT (GPT-3.5-Turbo) and Prompt-Engineering GPT-4

### Environment (OS: Windows or Linux)

```bash
pip install openai
pip install pandas
```
Note: The fine-tuning code has been slightly different as the version of openai updated to v1.0.0+.

Here, we provide the latest code.

### Implementation

Specific scripts  for each task are in the corresponding folders.

All notebooks of fine-tuning and prompt engineering GPTs (GPT-4, GPT-3.5) as well as evaluating for each task will be released soon...

###  Demo of Fine-tuning ChatGPT on small dataset

Here, we gave an example notebook of fine-tuning ChatGPT on 25 Paragraph2NMR data in ```demo/fine-tuning_chatgpt_on_25_paragraph2NMR_data.ipynb```, including:

 - Preprocessing
 - Training
 - Inferencing
 - Evaluating

## Fine-tuning Open-source Language Models 
## (Llama3, Llama2, Mistral, Bart, T5) 

### Environment (Linux)
```bash
mamba create -n llm python=3.10
mamba activate llm 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.1.2 numpy transformers==4.38.2 datasets tiktoken wandb tqdm
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple openpyxl
pip install -U -i https://pypi.tuna.tsinghua.edu.cn/simple peft==0.8.0 accelerate bitsandbytes safetensors
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple jsonlines
pip install -U -i https://pypi.tuna.tsinghua.edu.cn/simple vllm==0.3.1
pip install -U -i https://pypi.tuna.tsinghua.edu.cn/simple trl==0.7
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboardX
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboard
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple textdistance
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple nltk
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib
pip install -U -i https://pypi.tuna.tsinghua.edu.cn/simple ipywidgets
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple seaborn
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple seqeval
```

### Pretrained Models Downloads

Open-sourced pretrained models (Llama3, Llama2, Mistral, Bart, T5) can be downloaded from [huggingface](https://huggingface.co/models) or [modelscope](https://www.modelscope.cn/models).

### Fine-tuning

The codes and tutorials of Fine-tuning Language models (ChatGPT, Llama3, Llama2, Mistral, Bart, T5) for each task are in the corresponding folders.
