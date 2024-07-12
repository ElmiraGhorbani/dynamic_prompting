# read the contents of your README file
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

project_urls = {
    'Github': 'https://github.com/ElmiraGhorbani/dynamic_prompting',
}

setup(
    name='dynamic_prompting',
    version='0.1.1',
    author='Elmira Ghorbani',
    description="Dynamic Few-Shot Prompting is a Python package that dynamically selects N samples that are contextually close to the user's task or query from a knowledge base (similar to RAG) to include in the prompt.",
    packages=find_packages(),
    install_requires=[
        'einops==0.8.0',
        'sentence-transformers==3.0.1',
        'torch==2.2.1',
        'transformers==4.41.1',
        'llama3'
    ],
    dependency_links=[
        'git+https://github.com/meta-llama/llama3.git#egg=llama3'
    ],
    project_urls=project_urls,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
