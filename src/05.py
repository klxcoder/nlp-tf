import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat',
]

tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index) # {'i': 1, 'love': 2, 'my': 3, 'dog': 4, 'cat': 5}
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences) # [[1, 2, 3, 4], [1, 2, 3, 5]]