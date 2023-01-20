from random import randint

import requests
from datasets import load_dataset
from sklearn.metrics import accuracy_score

MODEL_NAME = "gpt-sw3-v2-40b"
N_SHOTS = 1

dataset = load_dataset("AI-Sweden/SuperLim", 'sweana')['test']
df = dataset.to_pandas()[0:20]
few_shots = df.sample(n=N_SHOTS, random_state=0)
eval_df = df.drop(few_shots.index)
predictions, labels, binary_results, binary_random_results = [], [], [], []

prompt = ""

for idx, row in few_shots.iterrows():
    prompt += f"{row['a']} - {row['b']} + {row['c']} = {row['d']}\n"

for idx, row in eval_df.iterrows():
    label = row['d']
    post_prompt = f"{row['a']} - {row['b']} + {row['c']} ="
    prompt_extended = prompt + post_prompt

    json_post = {
        "prompt": prompt_extended,
        "model": "gpt-sw3-v2-40b",
        "max_tokens": 128,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "stream": False,
        "no_repeat_ngram_size": 0,
        "repetition_penalty": 0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": "\n",
        "auth_token": "80c83dd62a842ddf"
    }

    response = requests.post(url="http://relay.aiqu.ai:33165/v1/engines/gpt-sw3/completions", json=json_post)
    pred = response.json()['choices'][0]['text'].strip()

    if pred == label:
        binary_results.append(1)
    else:
        binary_results.append(0)

    labels.append(label), predictions.append(pred), binary_random_results.append(randint(0, 1))

eval_df['labels'] = labels
eval_df['predictions'] = predictions
eval_df['binary_results'] = binary_results
eval_df['binary_random_results'] = binary_random_results

eval_df.to_csv(f'dataframes/sweana_{MODEL_NAME}_n_shots_{N_SHOTS}.csv', index=False)

with open(f'results/sweana_{MODEL_NAME}_n_shots_{N_SHOTS}.txt', 'w') as outfile:
    outfile.write(f'accuracy manual: {round(sum(binary_results) / len(binary_results), 4)}\n')
    outfile.write(f'accuracy sklearn: {round(accuracy_score(y_true=labels, y_pred=predictions), 4)}\n\n')
    outfile.write(f'N_SHOTS: {N_SHOTS}\n')
    outfile.write(f'prompt: {prompt}\n')
