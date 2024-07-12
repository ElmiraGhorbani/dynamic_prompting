# Llama3 Model

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


# Text embedding models

The Embeddings class is designed for interfacing with text embedding models. There are many embedding model providers (OpenAI, Cohere, Hugging Face, etc.). Currently, this class provides a standard interface for **mxbai-embed-large-v1**, **bge-small-en-v1.5**, and **nomic-embed-text-v1.5**.

# Get started

To start, download the models and put them in this folder.

```

git lfs install

```

### Nomic Model
```

git clone https://huggingface.co/nomic-ai/nomic-embed-text-v1.5

cd nomic-embed-text-v1.5

git lfs fetch

```

### BGE Model
```

git clone https://huggingface.co/BAAI/bge-small-en-v1.5

cd bge-small-en-v1.5

git lfs fetch

```


### Mxbai Model
```

git clone https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1

cd mxbai-embed-large-v1

git lfs fetch

```