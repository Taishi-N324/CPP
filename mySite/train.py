'''
必要に応じて !pip install
'''
from __future__ import absolute_import, division, print_function, unicode_literals

# try:
#   # %tensorflow_version only exists in Colab.
#   %tensorflow_version 2.x
# except Exception:
#   pass
import sys
import datetime, os
import tensorflow as tf
import string
import re
import csv
import numpy as np
import random
import pickle
import torch
from tensorflow import keras
from tensorflow.keras import layers
#from google.colab import files
import tensorflow as tf
#IS_COLAB = "google.colab" in sys.modules
# %load_ext tensorboard


#データ数増やす場合はgoogle driveに
#csvかtxt..etc..


name_csv = "ja_en/ja-en.csv"

ja_train_text = []
en_train_text = []
train_pairs = []
val_pairs = []
test_pairs = []

with open(name_csv, newline='') as csvfile:
  reader = csv.DictReader(csvfile)

  for row in reader:
    ja_train_text.append(row['Japanese'])
    en_train_text.append(row['English'])
    row['English'] = "\\" + " " + row['English'] +"|||"
    #ここの感覚は調整する
    #trainとval　を同じデータ、シャッフルするだけにしたらどうなるのか

    #学習データ：検証データ：テストデータ=8,1,1がよいらしいので。
    train_pairs.append((row['Japanese'],row['English']))
    random.shuffle(train_pairs)
  #print(len(train_pairs))

  for i in range(len(train_pairs)//10):
    val_pairs.append(train_pairs.pop(0))
    test_pairs.append(train_pairs[i])
  #print(len(train_pairs))


#あとで数値をいじる
vocab_size = 10000
sequence_length = 20

vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)

vectorization.adapt(ja_train_text)
vectorization.adapt(en_train_text)
#print(ja_train_text)
#print(en_train_text)

#データセットの件数が数百件程度であれば32, 64、数万件程度であれば1024, 2048　でtry


batch_size = 128

def format_dataset(ja, en):
    ja = vectorization(ja)
    en = vectorization(en)
    #ここでスライスするかどうかで変えられる
    '''
    japanese"、"english"にしないとエラー
    '''
    return ({"japanese": ja,
        "english": en,
    }, en)


def make_dataset(pairs):
    ja_texts, en_texts = zip(*pairs)
    ja_texts = list(ja_texts)
    en_texts = list(en_texts)
    dataset = tf.data.Dataset.from_tensor_slices((ja_texts, en_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache()

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)


embed_dim = 256
latent_dim = 1024

source = keras.Input(shape=(None,), dtype="int64", name="japanese")

x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(source)
encoded_source = layers.Bidirectional(
    layers.GRU(latent_dim), merge_mode="sum")(x)

past_target = keras.Input(shape=(None,), dtype="int64", name="english")
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(past_target)
decoder_gru = layers.GRU(latent_dim, return_sequences=True)
x = decoder_gru(x, initial_state=encoded_source)
x = layers.Dropout(0.5)(x)
target_next_step = layers.Dense(vocab_size, activation="softmax")(x)
seq2seq_rnn = keras.Model([source, past_target], target_next_step)


for inputs, targets in train_ds.take(1):
    print(f"inputs['japanese'].shape: {inputs['japanese'].shape}")
    print(f"inputs['english'].shape: {inputs['english'].shape}")
    print(f"targets.shape: {targets.shape}")

#format_datasetのところで調整可能


#RNNでseq2seqで訓練

seq2seq_rnn.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# %tensorboard --logdir logs/fit


#lossの収束は
#最初3 +10




#損失関数の値が収束するような、epochs数に調整を行う。
log = seq2seq_rnn.fit(train_ds,
                      epochs=1,
                      validation_data=val_ds,
                      callbacks=[tensorboard_callback])


'''
#原因解明をする


vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)

ここか
したのプログラム
'''


eng_vocab = vectorization.get_vocabulary()
eng_index_lookup = dict(zip(range(len(eng_vocab)), eng_vocab))
max_decoded_sentence_length = 15

def decode_sequence(input_sentence):
  #ここでエラーしてるな
    tokenized_input_sentence = vectorization([input_sentence])
    #print(tokenized_input_sentence)
    decoded_sentence = "\\ "
    #print(decoded_sentence)
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = vectorization([decoded_sentence])
        #print(tokenized_target_sentence)
        #print(tokenized_target_sentence)
        next_token_predictions = seq2seq_rnn.predict(
            [tokenized_input_sentence, tokenized_target_sentence])
        #print(next_token_predictions)
        sampled_token_index = np.argmax(next_token_predictions[0, i, :])
        #print(sampled_token_index)
        sampled_token = eng_index_lookup[sampled_token_index]
        #print(sampled_token)
        decoded_sentence += " " + sampled_token
        #print(decoded_sentence)
        if sampled_token == "|||":
            break
    return decoded_sentence

test_eng_texts = [pair[0] for pair in test_pairs]
#for _ in range(5):
#input_sentence = random.choice(test_eng_texts)
input_sentence = "おはようございます"
print("-")
print(input_sentence)
print(decode_sequence(input_sentence))
