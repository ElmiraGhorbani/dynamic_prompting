import pandas as pd
from dynamic_prompting.embedings.config import EmbeddingsConfig
from dynamic_prompting.knowledge_base.config import KnowledgeBaseConfig
from dynamic_prompting.knowledge_base.knowledge_base import \
    KnowledgeBaseManagement
from dynamic_prompting.llms.config import LLMConfig, PromptConfig
from dynamic_prompting.llms.llm import LlamaModel
from dynamic_prompting.llms.prompt import PromptManagement
from dynamic_prompting.utils.utils import get_project_root

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
