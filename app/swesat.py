import numpy as np
import spacy
from datasets import load_dataset

nlp = spacy.load("sv_core_news_lg")

dataset = load_dataset("AI-Sweden/SuperLim", 'swesat')['test']
results, preds = [], []

for r in dataset:
    target_item = r['target_item']
    target_item_vec = nlp(target_item)
    answers = r['answer_1'], r['answer_2'], r['answer_3'], r['answer_4'], r['answer_5']
    sims, labels = [], []
    for e, answer in enumerate(answers):
        try:
            answer_text, answer_label = answer.split('/')
        except ValueError:
            print('labels not added correctly')
            answer_text, answer_label = answer, 0
            # {'target_item': 'dedikerad', 'answer_1': 'stödjande/0', 'answer_2': 'beslutsam', ...'}
        if int(answer_label) == 1:
            ground_truth = e

        sims.append(target_item_vec.similarity(nlp(answer_text)))

    sims = np.asarray(sims)
    if ground_truth == sims.argmax():  # randint(0, 4):
        results.append(1)
    else:
        results.append(0)

print(f'nr of questions: {len(results)}')
print(f'nr of correct predictions: {sum(results)}')
print(f'random baseline accuracy', (len(results) / 5.0) / len(results))
print(f'word model accuracy:', round(sum(results) / len(results), 4))
