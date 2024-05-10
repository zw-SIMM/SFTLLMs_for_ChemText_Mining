# SFTLLMs_for_ChemText_Mining

## Download
```bash
git clone https://github.com/zw-SIMM/SFTLLMs_for_chemtext_mining
cd SFTLLMs_for_chemtext_mining
```

## 🖊 Datasets and Codes

Preprocessed data， fine-tuning codes， README workflows have been placed in corresponding folders:

- ```Paragraph2Comound/```

- ```Paragraph2RXNRole/prod/``` and ```Paragraph2RXNRole/role/```

- ```Paragraph2MOFInfo/```

- ```Paragraph2NMR/```

- ```Paragraph2Action/``` (dataset is derived from pistachio dataset, which is available upon request.)

## 💿Fine-tuning ChatGPT (GPT-3.5-Turbo) and Prompt-Engineering GPT-4

### Environment (OS: Windows or Linux)

```bash
pip install openai
pip install pandas
```
Note: The fine-tuning code has been slightly different as the version of openai updated to v1.0.0+.

Here, we provide the latest code.

### Implementation

Specific scripts  for each task are in the corresponding folders.

All notebooks of fine-tuning and prompt engineering GPTs (GPT-4, GPT-3.5) as well as evaluating for each task has beed released!

###  Demo of Fine-tuning ChatGPT on small dataset

Here, we gave an example notebook of fine-tuning ChatGPT on 25 Paragraph2NMR data in ```demo/fine-tuning_chatgpt_on_25_paragraph2NMR_data.ipynb```, including:

 - Preprocessing
 - Training
 - Inferencing
 - Evaluating

## 📀Fine-tuning Open-source Language Models (Mistral, Llama3, Bart, T5) 

### Environment (Linux)
```bash
mamba create -n llm python=3.10
mamba activate llm 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas numpy ipywidgets tqdm
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.1.2  transformers==4.38.2 datasets tiktoken wandb==0.11 openpyxl
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple peft==0.8.0 accelerate bitsandbytes safetensors jsonlines
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vllm==0.3.1
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple trl==0.7
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboardX tensorboard
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple textdistance nltk matplotlib seaborn seqeval
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope
```

### Pretrained Models Downloads

Open-sourced pretrained models (Llama3, Llama2, Mistral, Bart, T5) can be downloaded from [huggingface](https://huggingface.co/models) or [modelscope](https://www.modelscope.cn/models).

Here is an example for downloading pretrained models by scripts on linux servers from modelscope:
```python
from modelscope import snapshot_download
model_dir = snapshot_download("LLM-Research/Meta-Llama-3-8B-Instruct", revision='master', cache_dir='/home/pretrained_models')
model_dir = snapshot_download('AI-ModelScope/Mistral-7B-Instruct-v0.2', revision='master', cache_dir='/home/pretrained_models')
```

### Fine-tuning

The codes and tutorials of Fine-tuning Language models (ChatGPT, Llama3, Llama2, Mistral, Bart, T5) for each task are in the corresponding folders.
