print("ロードに時間がかかります")
import random
import tensorflow as tf
import re
from tensorflow import keras
import numpy as np
import random
import string
from tensorflow.keras import layers
from tensorflow.python.keras.models import load_model
from tensorflow import keras
vocab_size = 20000
sequence_length = 30
batch_size = 64
embed_dim = 256
latent_dim = 1024

train_japanese_texts = []
train_english_texts = []
with open("english_texts_pp") as f:
    datalist = f.readlines()
    for data in datalist:
        train_english_texts.append(data.strip())
with open("japanese_texts_pp") as f:
    datalist = f.readlines()
    for data in datalist:
        train_japanese_texts.append(data.strip())

#with tf.Session() as sess:
#     sess.run(tf.tables_initializer())
     #sess.run(dataset_iter.initializer)
     #data = sess.run(next_element)
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")

source_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
target_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)


strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")


source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_japanese_texts)



ja_vocab = target_vectorization.get_vocabulary()
ja_index_lookup = dict(zip(range(len(ja_vocab)), ja_vocab))
max_decoded_sentence_length = 20

model = keras.models.load_model('seq2seq_rnn.h5')
def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
     # print(tokenized_input_sentence)
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])
        #next_token_predictions = model.predict(
        next_token_predictions = model.predict(
            [tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(next_token_predictions[0, i, :])
        sampled_token = ja_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break

    return decoded_sentence



print("start-------------------")
while True:
    input_sentence = str(input())
    print(decode_sequence(input_sentence)[8:-5])
    print("----")
