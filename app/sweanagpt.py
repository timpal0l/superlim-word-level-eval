import requests
from datasets import load_dataset

dataset = load_dataset("AI-Sweden/SuperLim", 'sweana')['test']
df_all = dataset.to_pandas()
n_shots = [1, 3, 5, 10, 25, 50, 100]
df_train_pool = df_all.sample(n=max(n_shots), random_state=0)
eval_df = df_all.drop(df_train_pool.index)

models = [
    "gpt-sw3-126m",
    "gpt-sw3-356m",
    "gpt-sw3-1.3b",
    "gpt-sw3-6.7b",
    "gpt-sw3-20b"
]

for model in models[:1]:
    print(f'INFO:     running model: {model}')
    for shot in n_shots:
        print(f'INFO:     n shot: {shot}')
        few_shots = df_train_pool.sample(n=shot, random_state=0)
        predictions, labels, binary_results = [], [], []
        prompt = ""

        for idx, row in few_shots.iterrows():
            prompt += f"{row['a']} - {row['b']} + {row['c']} = {row['d']}\n"

        for idx, row in eval_df.iterrows():
            print(f'INFO:     running eval with model {model} and shots {shot}')
            label = row['d']
            post_prompt = f"{row['a']} - {row['b']} + {row['c']} ="
            prompt_extended = prompt + post_prompt

            json_post = {
                "prompt": prompt_extended,
                "model": model,
                "max_tokens": 64,
                "temperature": 0,
                "top_p": 1,
                "n": 1,
                "stream": False,
                "no_repeat_ngram_size": 0,
                "repetition_penalty": 0,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": "\n",
                "token": "y8ABUHCeVNCVqPycPkEgaayXCawkuv94",
                "user": "nlu",
            }

            response = requests.post(url="http://relay.aiqu.ai:48683/v1/engines/gpt-sw3/completions", json=json_post)
            pred = response.json()['choices'][0]['text'].strip()

            if pred == label:
                binary_results.append(1)
            else:
                binary_results.append(0)

            labels.append(label), predictions.append(pred)

        eval_df['labels'] = labels
        eval_df['predictions'] = predictions
        eval_df['binary_results'] = binary_results
        eval_df.to_csv(f'dataframes/sweana_{model}_n_shots_{shot}.csv', index=False)

"""
with open(f'results/sweana_{model}_n_shots_{shot}.txt', 'w') as outfile:
    outfile.write(f'accuracy sklearn: {round(accuracy_score(y_true=labels, y_pred=predictions), 4)}\n')
    outfile.write(f'N_SHOTS: {shot}\n')
    outfile.write(f'N_EVALS: {len(eval_df)}\n\n')
    outfile.write(f'prompt: {prompt}\n')
"""
