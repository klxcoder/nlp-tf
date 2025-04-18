import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?',
]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
# {'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
# [[5, 3, 2, 4], [5, 3, 2, 7], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]
padded = pad_sequences(sequences, padding='post', maxlen=5, truncating='post')
print(padded)
"""
[[5 3 2 4 0]
 [5 3 2 7 0]
 [6 3 2 4 0]
 [8 6 9 2 4]]
"""

test_data = [
    'i really love my dog',
    'my dog loves my manatee',
]

test_sequences = tokenizer.texts_to_sequences(test_data)
print(test_sequences)
# [[5, 1, 3, 2, 4], [2, 4, 1, 2, 1]]