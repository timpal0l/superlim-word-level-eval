import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda:5"

models = [
    "/data/models/AI-Sweden/gpt-sw3-126m",
    "/data/models/AI-Sweden/gpt-sw3-356m",
    "/data/models/AI-Sweden/gpt-sw3-1.3b",
    "tiiuae/falcon-rw-1b",
    "/data/models/AI-Sweden/gpt-sw3-6.7b-v2",
    "tiiuae/falcon-7b",
    "/data/models/AI-Sweden/gpt-sw3-20b",
    "EleutherAI/gpt-neox-20b",
    "/data/models/AI-Sweden/gpt-sw3-40b",
    "tiiuae/falcon-40b",
]
langs = ['en', 'sv', 'no', 'da', 'fi']

# Create an empty DataFrame with the required columns
results = pd.DataFrame(columns=["language", "model", "ppl"])

for lang in langs:
    text = '\n\n'.join(pd.read_csv(f"/data/datasets/scandisent/{lang}.csv")['text'])
    Z = len(text)

    for model_name in models:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True,
            trust_remote_code=True
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encodings = tokenizer(text, return_tensors="pt")

        max_length = 2048  # model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        total_char_length = 0

        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss.to(torch.float32) * (trg_len - 1) / Z
                neg_log_likelihood = neg_log_likelihood.to(device)  # Move tensor to the right device

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        nll = torch.stack(nlls).sum()
        ppl = torch.exp(nll)
        print(model_name, round(ppl.item(), 4))

        # Append the new row to the DataFrame
        new_row = pd.DataFrame({"language": [lang], "model": [model_name], "ppl": [round(ppl.item(), 4)]})
        results = pd.concat([results, new_row], ignore_index=True)

        # Check if the file exists and if it is not empty (i.e., it has a header)
        filename = "results.csv"
        if not os.path.isfile(filename) or os.stat(filename).st_size == 0:
            new_row.to_csv(filename, index=False)
        else:  # Otherwise it exists, append without writing the header
            new_row.to_csv(filename, mode='a', header=False, index=False)

    print(results)
