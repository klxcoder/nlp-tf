# https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection
import json

count = 0

sentences = []
labels = []
urls = []

with open('./dataset/Sarcasm_Headlines_Dataset.json', 'r') as f:
    for line in f:
        item = json.loads(line.strip())
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])
        urls.append(item['article_link'])
        count += 1
        if count==2:
            break

print(sentences)
# ["former versace store clerk sues over secret 'black code' for minority shoppers", "the 'roseanne' revival catches up to our thorny political mood, for better and worse"]

print(labels)
# [0, 0]

print(urls)
# ['https://www.huffingtonpost.com/entry/versace-black-code_us_5861fbefe4b0de3a08f600d5', 'https://www.huffingtonpost.com/entry/roseanne-revival-review_us_5ab3a497e4b054d118e04365']
