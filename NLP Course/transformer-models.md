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

# Encoder Models

# Decoder Models

# Sequence-to-sequence Models

# Bias and Limitations

# Summary
