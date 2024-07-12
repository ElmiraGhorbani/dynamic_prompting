import os
import warnings
from textwrap import dedent
from typing import List

import torch
import torch.distributed as dist
import transformers
from dynamic_prompting.llms.config import LLMConfig
from dynamic_prompting.utils.utils import get_project_root
from llama import Dialog, Llama

warnings.filterwarnings('ignore')


class LlamaModel:
    def __init__(self, llm_config: LLMConfig):
        """
        Initializes the LlamaModel instance.

        Args:
            rank (int): Rank of the current process in distributed training.
            world_size (int): Total number of processes in distributed training.
            max_seq_len (int): Maximum sequence length for the model.
            max_batch_size (int): Maximum batch size for the model.
        """
        super().__init__()
        self.config = llm_config

        self.root_path = get_project_root()
        self.generator = self.load_model()

    def setup(self):
        """
        Sets up the distributed environment for PyTorch Distributed Data Parallelism.
        """
        os.environ["MASTER_ADDR"] = 'localhost'
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=self.config.rank,
                                world_size=self.config.world_size)

    def cleanup(self):
        """
        Cleans up the distributed environment.
        """
        dist.destroy_process_group()

    def load_model(self):
        """
        Loads the Llama model using the specified checkpoint directory and tokenizer.

        Returns:
            Llama: Initialized Llama model.
        """
        if self.config.local_files:
            ckpt_dir = f'{self.root_path }/models/Meta-Llama-3-8B-Instruct'
            if os.path.exists(ckpt_dir):
                assert FileNotFoundError(
                    'Visit the Meta Llama website(https://llama.meta.com/llama-downloads) and register to download the model/s.')

            self.setup()
            generator = Llama.build(
                ckpt_dir=ckpt_dir,
                tokenizer_path=f'{ckpt_dir}/tokenizer.model',
                max_seq_len=self.config.max_seq_len,
                max_batch_size=self.config.max_batch_size,
            )
            return generator
        else:
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
            return pipeline

    def inference(
            self,
            system_instruction: str = '',
            user_input: str = '',
            max_gen_len: int = 1024,
            temperature: float = 0.7,
            top_p: float = 1.0):
        """
        Performs inference using the loaded Llama model.

        Args:
            system_instruction (str): System instruction text.
            user_input (str): User input text.
            max_gen_len (int): Maximum length of generated output.
            temperature (float): Sampling temperature for generation.
            top_p (float): Top-p nucleus sampling parameter.

        Returns:
            str: Generated response from the model.
        """
        if self.config.local_files:
            if self.generator is None:
                raise RuntimeError(
                    "Model has not been loaded. Call load_model first.")

            system_instruction = dedent(system_instruction)
            user_input = dedent(user_input)

            dialogs: List[Dialog] = [
                [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_input}
                ],
            ]

            results = self.generator.chat_completion(
                dialogs,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            return results[0]['generation']['content']
        else:
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_input},
            ]

            terminators = [
                self.generator.tokenizer.eos_token_id,
                self.generator.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = self.generator(
                messages,
                max_new_tokens=max_gen_len,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            return outputs[0]["generated_text"][-1]
