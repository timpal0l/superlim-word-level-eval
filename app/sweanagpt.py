import requests
from datasets import load_dataset

dataset = load_dataset("sbx/superlim-2", 'sweana')['train']
df_all = dataset.to_pandas()
n_shots = [175]
df_train_pool = df_all.sample(n=max(n_shots), random_state=0)
eval_df = load_dataset("sbx/superlim-2", 'sweana')['test'].to_pandas()

models = [
    "gpt-sw3-126m",
    "gpt-sw3-40b",
]

for model in models[:1]:
    print(f'INFO:     running model: {model}')
    for shot in n_shots:
        print(f'INFO:     n shot: {shot}')
        few_shots = df_train_pool.sample(n=shot, random_state=0)
        predictions, labels, binary_results = [], [], []
        prompt = ""

        for idx, row in few_shots.iterrows():
            prompt += f"{row['pair1_element1']} - {row['pair1_element2']} + {row['pair2_element1']} = {row['label']}\n"

        for idx, row in eval_df.iterrows():
            print(f'INFO:     running eval with model {model} and shots {shot}')
            label = row['label']
            post_prompt = f"{row['pair1_element1']} - {row['pair1_element2']} + {row['pair2_element1']} ="
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

            response = requests.post(url="https://gpt.ai.se/v1/engines/gpt-sw3/completions", json=json_post)

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

"""
for file in files:
    model_name = file.split('_')[1]
    n_shots = file.split('_')[-1].split('.')[0]
    df = pd.read_csv(file)
    accuracy = round(df.binary_results.sum() / df.shape[0], 4)
    print(f"model_name: {model_name}, n_shots: {n_shots}, accuracy: {accuracy}")
"""
