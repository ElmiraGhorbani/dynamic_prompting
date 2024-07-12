# DynamicPrompting

Dynamic Few-Shot Prompting is a Python package that dynamically selects N samples that are contextually close to the user's task or query from a knowledge base (similar to RAG) to include in the prompt. This allows the provision of related and informative examples to the LLM without needing to fine-tune the LLM to learn and extract patterns from the dataset. This means lower computation costs and resources; you just need to collect samples that are related to your task and might not be very clear for the LLM.

## Key Features

- Faster than fine-tuning a LLM
- Better performance than static few-shot prompts
- Works with less data than fine-tuning a model

## Quick Start

With pip:

```
pip install dynamic_prompting
```
or install from source

```
git clone https://github.com/ElmiraGhorbani/dynamic_prompting

pip install -e
```

## Inference

### First setup the model:

#### Llama3 Model

1) Visit the Meta Llama website(https://llama.meta.com/llama-downloads) and register to download the model/s.

2) Once registered, you will get an email with a URL to download the models. You will need this URL when you run the download.sh script.

3) Once you get the email, navigate to your downloaded llama repository and run the download.sh script.

    - Make sure to grant execution permissions to the download.sh script
    - During this process, you will be prompted to enter the URL from the email.
    - Do not use the “Copy Link” option; copy the link from the email manually.

**Once you have downloaded the models, put them in this folder.**

**otherwise you can use this way:**

    - Step 1: Go to your Hugging Face account “Settings” and then “Access Tokens” on the left column, and copy the token you need.

    - Step 2: On your terminal, export your token starting with “HF_”. Use a distinct name (for example, HF_TOKEN) for each token you export. 
    You may add this line to your ~/.bashrc if you do not want to export it every time you start a session.


```
export HF_TOKEN="HF_XXXXXXXXXXXXX"
```


#### Text embedding models

The Embeddings class is designed for interfacing with text embedding models. There are many embedding model providers (OpenAI, Cohere, Hugging Face, etc.). Currently, this class provides a standard interface for **mxbai-embed-large-v1**, **bge-small-en-v1.5**, and **nomic-embed-text-v1.5**.

##### Get started

To start, download the models and put them in this folder.

```

git lfs install

```

###### Nomic Model
```

git clone https://huggingface.co/nomic-ai/nomic-embed-text-v1.5

cd nomic-embed-text-v1.5

git lfs fetch

```

###### BGE Model
```

git clone https://huggingface.co/BAAI/bge-small-en-v1.5

cd bge-small-en-v1.5

git lfs fetch

```


###### Mxbai Model
```

git clone https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1

cd mxbai-embed-large-v1

git lfs fetch

```

###### Import the necessary packages:

 ```python
import pandas as pd
from dynamic_prompting.embedings.config import EmbeddingsConfig
from dynamic_prompting.knowledge_base.config import KnowledgeBaseConfig
from dynamic_prompting.knowledge_base.knowledge_base import \
    KnowledgeBaseManagement
from dynamic_prompting.llms.config import LLMConfig, PromptConfig
from dynamic_prompting.llms.llm import LlamaModel
from dynamic_prompting.llms.prompt import PromptManagement
from dynamic_prompting.utils.utils import get_project_root
 ```

###### Set configs

```python
emb_config = EmbeddingsConfig(
    model_name="nomic-embed-text-v1.5",
    local_files=True,
    trust_remote_code=True,
    max_dimension=512,
)
kb_config = KnowledgeBaseConfig(
    knowledge_base_name='nomic_signal_classification2')

llm_config = LLMConfig(
    rank=0,
    world_size=1,
    max_seq_len=1024,
    max_batch_size=8,
    local_files=True
)

prompt = """
### Instruction:
You are given a text sample, and your task is to classify it into one of the following categories: Business, Science, Sports,\n
Politics, Entertainment, Weather, Technology, Health, Local, World, Culture, Education, Travel.\n
Carefully read the text and determine the most appropriate category that best describes the main topic of the text.\n

Examples:
{examples}

**Text:** "Your text sample here"
**Category:** "Only return the category"
"""
prompt_config = PromptConfig(prompt=prompt)

```

Use the dataset sample to run inference(**classification task**):

```python
kb = KnowledgeBaseManagement(embeddings_config=emb_config, kb_config=kb_config)

llama_model = LlamaModel(llm_config=llm_config)


prompt_handler = PromptManagement(prompt_config=prompt_config)

root_path = get_project_root()


df = pd.read_csv(
    f'{root_path}/data/sample.csv',
    index_col=False
)
messages = df['txt'].tolist()
messages = [i.replace('\n', ' ') for i in messages]
labels = df['label'].tolist()

kb.create_kb(messages)
db = kb.load_kb()

query = '''The city council has announced a new initiative to improve public transportation\
    infrastructure, aiming to reduce traffic congestion and promote eco-friendly travel options.'''

indx = kb.search_kb(
    query=query,
    embeddings=db,
    num_of_neighbours=5
)


samples = [messages[i] for i in indx]
labels = [labels[i] for i in indx]

prompt = prompt_handler.set_few_shots(context_input=samples, labels=labels)

# run model prediction
inf_result = llama_model.inference(
    system_instruction=prompt,
    user_input=query
)

print(inf_result)
```