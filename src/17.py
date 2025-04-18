# https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

# count = 0

sentences = []
labels = []
urls = []

with open('./dataset/Sarcasm_Headlines_Dataset.json', 'r') as f:
    for line in f:
        item = json.loads(line.strip())
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])
        urls.append(item['article_link'])
        # count += 1
        # if count==10:
        #     break

# training_size = 8

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
# Filter out tokens >= num_words
training_sequences = [[token for token in seq if token < vocab_size] for seq in training_sequences]
training_padded = pad_sequences(training_sequences)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_sequences = [[token for token in seq if token < vocab_size] for seq in testing_sequences]
testing_padded = pad_sequences(testing_sequences)

assert training_padded.max() < vocab_size

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 30

history = model.fit(
    np.array(training_padded),
    np.array(training_labels),
    epochs=num_epochs,
    validation_data=(np.array(testing_padded), np.array(testing_labels)),
    verbose=2,
)

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

# plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")

sentence = [
    "granny starting to fear spiders in the garden might be real",
    "game of thrones season finale showing this sunday night",
]
sequences = tokenizer.texts_to_sequences(sentence)
sequences = [[token for token in seq if token < vocab_size] for seq in sequences]
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))