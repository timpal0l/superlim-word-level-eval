import spacy
from datasets import load_dataset
from scipy import spatial

nlp = spacy.load("sv_core_news_lg")

dataset = load_dataset("AI-Sweden/SuperLim", 'sweana')['test']
cosine_similarity = lambda vec1, vec2: 1 - spatial.distance.cosine(vec1, vec2)

sims_pred, sims_label = [], []
for row in dataset:
    w1 = nlp(row['a'])
    w2 = nlp(row['b'])
    w3 = nlp(row['c'])

    w4 = nlp(row['d'])
    wx = w1.vector - w2.vector + w3.vector

    similarities = []

    for word in nlp.vocab:
        if word.has_vector and word.is_alpha and word.is_lower:
            similarities.append((cosine_similarity(wx, word.vector), word.text))

    print(sorted(similarities, reverse=True)[:10], '---target', w4)

