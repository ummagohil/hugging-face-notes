# Training a new tokeniser from an old one

- Training a tokeniser is not the same as training a model
- Model training uses stochastic gradient descent, tokeniser training uses a statistical process which is deterministic

- An inefficient way:

```python
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

- A better approach:

```python
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

```

# Fast tokenisers' special powers

- It's only when tokenising lots of texts in parallel tat the same time that oyu will be able to see the difference

```python
from transformers import pipeline

token_classifier = pipeline("token-classification", aggregation_strategy="simple")
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

```python
results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        start, end = offsets[idx]
        results.append(
            {
                "entity": label,
                "score": probabilities[idx][pred],
                "word": tokens[idx],
                "start": start,
                "end": end,
            }
        )

print(results)
```

# Fast tokenisers in the QA pipeline

- Using a model for question answering:

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)
```

# Normalisation and pre-tokenisation

```python
tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
```

# Building a tokeniser, block by block

- Normalisation -> pre-tokenisation -> model -> postprocessor
- Tokenisers have the following building blocks
  - normalizers contains all the possible types of Normalizer
  - pre_tokenizers contains all the possible types of PreTokenizer you can use
  - models contains the various types of Model you can use, like BPE, WordPiece, and Unigram
  - trainers contains all the different types of Trainer you can use to train your model on a corpus (one per type of model)
  - post_processors contains the various types of PostProcessor you can use
  - decoders contains the various types of Decoder you can use to decode the outputs of tokenization
- To trainer a new tokeniser, need to use a small corpus of text so the examples run fast - like WikiText-2
