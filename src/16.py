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
        if count==10:
            break

training_size = 8
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# print(json.dumps(training_sentences, indent=2))
"""
[
  "former versace store clerk sues over secret 'black code' for minority shoppers",
  "the 'roseanne' revival catches up to our thorny political mood, for better and worse",
  "mom starting to fear son's web series closest thing she will have to grandchild",
  "boehner just wants wife to listen, not come up with alternative debt-reduction ideas",
  "j.k. rowling wishes snape happy birthday in the most magical way",
  "advancing the world's women",
  "the fascinating case for eating lab-grown meat",
  "this ceo will send your kids to school, if you work for his company"
]
"""

# print(json.dumps(testing_sentences, indent=2))
"""
[
  "top snake handler leaves sinking huckabee campaign",
  "friday's morning email: inside trump's presser for the ages"
]
"""

# print(training_labels) # [0, 0, 1, 1, 0, 0, 0, 0]

# print(testing_labels) # [1, 0]

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences)

print(testing_sequences) # [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 3, 4, 1]]

print(testing_padded)
"""
[[0 0 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 3 4 1]]
"""
