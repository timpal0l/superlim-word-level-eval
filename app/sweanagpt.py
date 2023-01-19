import requests
from datasets import load_dataset

dataset = load_dataset("AI-Sweden/SuperLim", 'sweana')['test']

prompt = """
skjuter - sköt + bleker = blekte\n
örn - örnar + fågel = fåglar\n
Vallentuna - Stockholm + Ekerö = Stockholm\n
rik - rikare + låg = lägre\n
Sofia - Bulgarien + Luanda = Angola\n
driver - drev + stänger = stängde\n
byter - bytte + åker = åkte\n
Hemse - Gotland + Nyköping = Södermanland\n
varm - varmast + djup = djupast\n
Libreville - Gabon + Asmara = Eritrea\n
Warszawa - Polen + Asmara = Eritrea\n
Minsk - Vitryssland + Asmara = Eritrea\n
Nyköping - Södermanland + Kalmar = Kalmar\n
hård - hårdare + grön = grönare\n
tung - tyngst + svag = svagast\n
Singapore - Singapore + Maseru = Lesotho\n
moralisk - omoralisk + låg = hög\n
Antananarivo - Madagaskar + Minsk = Vitryssland\n
telefon - telefoner + bok = böcker\n
Warszawa - Polen + Zagreb = Kroatien\n
Baku - Azerbajdzjan + Rabat = Marocko\n
Budapest - Ungern + Mogadishu = Somalia\n
dyr - billig + möjlig = omöjlig\n
Piteå - Norrbotten + Laholm = Halland\n
sko - skor + åsna = åsnor\n
skarp - skarpare + vit = vitare\n
tung - tyngre + grön = grönare\n
Bamako - Mali + Bern = Schweiz\n
grön - grönast + snabb = snabbast\n
låg - hög + kort = lång\n
snabb - långsam + kort = lång\n
Dublin - Irland + Ljubljana = Slovenien\n
tung - tyngre + bra = bättre\n
Rabat - Marocko + Katmandu = Nepal\n
bok - böcker + fågel = fåglar\n
Reykjavik - Island + Kabul = Afghanistan\n
tar - tog + måste = måste\n
svår - enkel + möjlig = omöjlig\n
Bratislava - Slovakien + Alger = Algeriet\n
smart - smartast + kylig = kyligast\n
Ljubljana - Slovenien + Gaborone = Botswana\n
Tasjkent - Uzbekistan + Belgrad = Serbien\n
Budapest - Ungern + Dusjanbe = Tadzjikistan\n
prins - prinsessa + son = dotter\n
tjock - tjockast + vit = vitast\n
Tirana - Albanien + Antananarivo = Madagaskar\n
Asmara - Eritrea + Peking = Kina\n
Manama - Bahrain + Dusjanbe = Tadzjikistan\n
blänker - blänkte + hugger = högg\n
Managua - Nicaragua + Ottawa = Kanada\n
"""

for row in dataset:
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

    if label == pred:
        print(f'Correct! {label} == {pred}')

    else:
        print(f'Incorrect! {label} != {pred}')
