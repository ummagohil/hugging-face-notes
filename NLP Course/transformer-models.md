# Natural Language Processing

The following are common NLP tasks:

- Classifying whole sentences
- Classifying each word in a sentence
- Generating text content
- Extracting an answer from text
- Generating a new sentence from an input text

# Transformers, what can they do?

- Transformer library provides the functionality to create and use shared models
- Model Hub contains many pre-trained models that anyone can use (and you can upload your own models too)
- The pipeline function returns an end-to-end object that performs an NLP task on one or several texts
- It connects a model with its necessary preprocessing and postprocessing steps, allowing us to directly input any text and get an intelligible answer
- In the code below, we try the pipeline API as sentiment analysis - it classifies text as positive or negative

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```

- The output will be as follows:

```python
[{'label': 'POSITIVE', 'score': 0.9598047137260437}]
```

- You can batch sentences as the following:

```python
classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
```

- The output should look like this:

```python
[{'label': 'POSITIVE', 'score': 0.9598047137260437},
 {'label': 'NEGATIVE', 'score': 0.9994558095932007}]
```

- The zero-shot-classification pipeline lets you select the labels for classification
- The text-generation pipeline uses an input prompt to generate text

- Here is an example of using the distilgpt2 model

```python
from transformers import pipeline
generator = pipeline("text-generation", model="distilgpt2")
generator(
  "In this course, we will teach you how to",
  max_length=30,
  num_return_sequences=2
)
```

Output:

```
[{'generated_text': 'In this course, we will teach you how to understand and use '
    'data flow and data interchange when handling user data. We '
    'will be working with one or more of the most commonly used '
    'data flows — data flows of various types, as seen by the '
    'HTTP'}]
```

- The fill-mask pipeline can predict missing words in a sentences
- The NER pipeline identifies entities such as person, organisations or locations in a sentence
- Translation pipeline translates text from one language to another

# How do Transformers work?

- Transfer learning is the act of initialising a model with another model's weights
- More efficient to fine tune a model than create a new one from raw data
- ImageNet is commonly used as a dataset for pretraining models in computer vision
- Transfer learning is applied by dropping the head of the pretrained model while keeping its body
- The transformer is based on the attention mechanism
- Encoders and decoders can be used together or separately
- Encoder "encodes" text into numerical representations (these numerical representations can also be called embeddings or features)
- The decoder "decodes" the representations from the encoder and has a unidirectional feature
  Encoder: The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.
  Decoder: The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.

# Encoder Models

- The encoder outputs a numerical representation for each word used as input
- The representation is made up of a vector of values for each word of the initial sequence
- Each word in the initial sequence affects every word's representation

## Why Should You Use an Encoder?

- Bi-directional: context from the left and the right
- Good at extracting meaningful information
- Sequence classification, question answering, masked language modelling
- NLU: Natural Language Understanding
- Example of encoders: BERT, RoBERTa, ALBERT

# Decoder Models

- The decoder creates a feature tensor from an initial sequence
- It outputs numerical representation from an initial sequence
- An example feature tensor is made up of a vector of values for each word of the initial sequence
- Words can only see the words on their left side, the right side is hidden
- The masked self-attention layer hides the values of context on the right
- Decoders, with the unidirectional context, are good at generating words given a context

## Why Should You Use a Decoder?

- Unidirectional: access to their left (or right) context
- Great at causal tasks - generating sequences
- NLG: Natural Language Generation
- Example of decoders: GTP-2, GPT Neo

# Sequence-to-sequence Models (Encoder-Decoder)

- Sequence to sequence tasks, many-to-many: translation, summarisation
- Weights are not necessarily shared across the encoder and decoder
- Input distribution is different from output distribution
- Used for things like translation - think Google Translate

> Google Translate uses Neural Machine Translation (NMT), primarily powered > by the Transformer architecture, for accurate and fluent translations. The system works as follows:
> Preprocessing: The input text is tokenized, language-detected, and normalized.
>
> Translation via NMT:
> An Encoder-Decoder framework processes input in the source language and generates output in the target language.
> Self-Attention Mechanism ensures the model captures context across sentences.
> Beam Search selects the most probable translation.
> Post-Processing: Detokenization and formatting ensure natural output.
>
> Key Technologies:
> Transformers: Efficiently handle context, replacing older models like RNNs.
> Multilingual Training: Enables translations between languages even without direct bilingual data.
> Custom Hardware: Google uses TPUs for efficient model training and inference.
>
> Benefits:
> Context-aware and fluent translations.
> Continuous improvement via user feedback and updated training data.
>
> Limitations:
> Struggles with idiomatic expressions, low-resource languages, and complex grammar.
> In essence, Google Translate is a state-of-the-art application of deep learning and NLP, designed for large-scale and efficient translation services.
