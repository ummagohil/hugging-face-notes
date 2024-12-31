# Behind the Pipeline

## PyTorch

- Tokenizer -> Model -> Post Processing

- Preprocessing with a tokenizer: convert text inputs into numbers that the model can understand
  - Splitting the input into words, subwords or symbols (eg. punctuation) that are called **tokens**
  - Mapping each token to an integer
  - Adding additional inputs that may be useful to the model

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

- Once the tokeniser has been initialised, can pass sentences and into it and get a dictionary for the model. Then you can convert the list of input IDs to **tensors**
- Transfomer models only accept tensors as input
- Another word for tensors is **NumPy**
- To specify the type of tensor you want to get back (PyTorch, TensorFlow or plain NumPy) can use the `return_tensors` arg

```python
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```

Output is a dictionary containing two keys, `input_ids` and `attention_mask`

```
{
  'input_ids': tensor([
      [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
      [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
  ]),
  'attention_mask': tensor([
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  ])
}
```

- Vector output by the Transfomer module is usually large and has three dimensions:

  - Batch size: the number of sequences processed at a time (in the above example we have two)
  - Sequence length: the length of the numerical representation of the sequence (16 in our example)
  - Hidden size: the vector dimension of each model input

- An example of sequence classification head (to be able to classify the sentences as positive or negative):

```python
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
```

- Dimensionality will be much lower, the model head takes an input the high dimensional vectors and outputs vectors containing two values (open per label) - this will give the shape of the outouts

```
print(outputs.logits.shape)
```

```
torch.Size([2,2])
```

- Postprocessing the output:

```
print(outputs.logits)
```

```
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)
```

- To be able to convert to probabilities need to go through a SoftMaz layer

```python
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```

```
tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
```

# Models

- First, we need to initialise a BERT model to create a transformer

```python
from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)
```

- Then we need to load a transformer model that has already been trained

```python
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```

- If you want to save the model: `model.save_pretrained("directory_on_my_computer")`
- If you want to run the model locally, you'll need the following commands:

```
ls directory_on_my_computer

config.json pytorch_model.bin
```

- Once you have set this up, you might want to add a tokenizer to be able to encode/decode the data to feed into the model.

```python
import torch

model_inputs = torch.tensor(encoded_sequences)
```

```
output = model(model_inputs)
```

# Tokenisers

- Text based:

```python
tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)
```

```
['Jim', 'Henson', 'was', 'a', 'puppeteer']
```

- Character based: splitting into chars such as ['w', 'a', 's',',''!']
- Subword: rely on frequency used works not being split into smaller subwords but rare words should be recomposed into meaningful subwords
  - "annoyingly" would be considered a rare word and can be decomposed into "annoing" and "ly"

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

```
tokenizer("Using a Transformer network is simple")
```

```
{'input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

# Handling Multiple Sequences

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)
```

```
Input IDs: [[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607, 2026,  2878,  2166,  1012]]
Logits: [[-2.7276,  2.8789]]
```

- **Padding** makes sure all our sentences have the same length by adding a special word called the padding token to the sentences with fewer values.

```python
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)
```

```
tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward>)
tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
tensor([[ 1.5694, -1.3895],
        [ 1.3373, -1.2163]], grad_fn=<AddmmBackward>)
```

- **Attention masks** are tensors with the exact same shape as the input IDs tensor, filled with 0s and 1s: 1s indicate the corresponding tokens should be attended to, and 0s indicate the corresponding tokens should not be attended to

```python
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
```

```
tensor([[ 1.5694, -1.3895],
        [ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
```

# Putting it all Together (From Tokenizer to Model)

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```

# Basic Usage Completed
