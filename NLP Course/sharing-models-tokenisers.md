# Using Pretrained Models

- Using the `Auto*` classes makes switching checkpoints easier

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")
```

# Sharing Pretrained Models

- Execute the following if in notebook to log into Hugging Face:

```
from huggingface_hub import notebook_login

notebook_login()
```

- Execute the following if on a command line to log into Hugging Face:

```
huggingface-cli login
```

- Once logged in, you can initialise transformers and models, and push these changes

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "camembert-base"

model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

```python
model.push_to_hub("dummy-model")
```

```python
tokenizer.push_to_hub("dummy-model")
```

## The `huggingface_hub` Python Library

- A package which offers a set of tools for the model and dataset hubs
- Provides methods and classes for common tasks such as repo info on the hub

```python
from huggingface_hub import (
    # User management
    login,
    logout,
    whoami,

    # Repository creation and management
    create_repo,
    delete_repo,
    update_repo_visibility,

    # And some methods to retrieve/change information about the content
    list_models,
    list_datasets,
    list_metrics,
    list_repo_files,
    upload_file,
    delete_file,
)
```

```python
from huggingface_hub import create_repo

create_repo("dummy-model")
```

## Uploading Files via `upload_file`

```python
from huggingface_hub import upload_file

upload_file(
    "<path_to_file>/config.json",
    path_in_repo="config.json",
    repo_id="<namespace>/dummy-model",
)
```

## The Repository Class

- Manages a local repo in a git like manner

```python
from huggingface_hub import Repository

repo = Repository("<path_to_dummy_folder>", clone_from="<namespace>/dummy-model")
```

- Can leverage several git methods:

```
repo.git_pull()
repo.git_add()
repo.git_commit()
repo.git_push()
repo.git_tag()
```

- An example set of commands:

```
repo.git_pull()
```

```python
model.save_pretrained("<path_to_dummy_folder>")
tokenizer.save_pretrained("<path_to_dummy_folder>")
```

```
repo.git_add()
repo.git_commit("Add model and tokenizer files")
repo.git_push()
```

# Building a Model Card

- Model card metadata:

```
---
language: fr
license: mit
datasets:
- oscar
---
```
