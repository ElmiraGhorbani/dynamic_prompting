from dynamic_prompting.llms.config import PromptConfig
from dynamic_prompting.utils.utils import get_project_root


class PromptManagement:
    def __init__(self, prompt_config: PromptConfig):
        """
        Initializes the PromptManagement object.
        Sets up the root path and reads the initial prompt from a file.
        """
        super().__init__()
        self.config = prompt_config
        self.root_path = get_project_root()
        self.prompt = self.config.prompt
    
    def set_few_shots(self, context_input: list, labels: list):
        """
        Creates a formatted string based on provided context input and labels.

        Args:
        - context_input (list): List of strings representing input contexts.
        - labels (list): List of strings representing corresponding output labels.

        Returns:
        - output_str (str): Formatted string with each context-input-label pair.
        """
        output_str = "\n".join(f"- Input: {input}, Output: {output}" for input, output in zip(context_input, labels))
        prompt = self.prompt.format(examples=output_str)
        return prompt
