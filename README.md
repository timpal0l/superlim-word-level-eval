# superlim-word-level-eval

A simple repo to evaluate the word-level tasks on [Superlim](https://huggingface.co/datasets/AI-Sweden/SuperLim).

## Installation

#### Install deps and download word embeddings for Swedish

```
pip install -r requirements.txt
python -m spacy download sv_core_news_lg
```

## Results

#### swesim_similarity

A) With no training, we can compare the cosine similarity between the word embeddings for `word_1` and `word_2` and
computing the pearson correlation coeff. Evaluation on the full test set.

| pearson    | spearman |
|------------|----------|
| 0.4336     | 0.5023   |

B) With training, we can add a linear layer to learn how "similarity" is expressed in this dataset. E.g 80/20 split
with 5 folds cv for a good measurement.

| pearson | spearman |
|---------|----------|
|         |          |

#### swesim_relatedness

A) With no training, we can compare the cosine similarity between the word embeddings for `word_1` and `word_2` and
computing the pearson correlation coeff. Evaluation on the full test set.

| pearson | spearman |
|---------|----------|
| 0.5171  | 0.5135   |

B) With training, we can add a linear layer to learn how "relatedness" is expressed in this dataset. E.g 80/20 split
with 5 folds cv for a good measurement.

##### swesat

With a target word and 5 suggested synonyms predict the correct synonym. There are 822 `target_items`, and 399 predictions
were correct.

| accuracy | random |
|----------|--------|
| 0.4854   | 0.2    |

#### sweana
