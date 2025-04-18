# https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# print(sentences)
# ["former versace store clerk sues over secret 'black code' for minority shoppers", "the 'roseanne' revival catches up to our thorny political mood, for better and worse"]

# print(labels)
# [0, 0]

# print(urls)
# ['https://www.huffingtonpost.com/entry/versace-black-code_us_5861fbefe4b0de3a08f600d5', 'https://www.huffingtonpost.com/entry/roseanne-revival-review_us_5ab3a497e4b054d118e04365']

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences)
print(padded[0]) # [ 0  0  3  4  5  6  7  8  9 10 11  2 12 13]
print(padded.shape) # (2, 14)