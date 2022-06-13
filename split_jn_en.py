#インストール
#!pip install mecab-python3

#辞書インストール
#!pip install unidic-lite

#! git clone https://github.com/Taishi-N324/CPP


import MeCab
import csv
import re
from tqdm import tqdm
import os

new_dir_path = 'data/'

os.mkdir(new_dir_path)


en_trains = []
ja_trains = []

tagger = MeCab.Tagger()
wakati = MeCab.Tagger("-Owakati")

with open('CPP/ja_en_250000_7.txt') as f:
    reader = csv.reader(f)
    i=0
    for row in tqdm(reader):
      texts = ""
      for text in row:
        texts += text

      ja_en = texts.split("\t")
      if(len(ja_en)) == 2:
        en_trains.append(ja_en[0])
        ja_trains.append(ja_en[1])



with open('data/en_train.txt', 'w') as f:
    for en_train in tqdm(en_trains):
      f.write(en_train)
      f.write('\n')
f.close()

with open('data/ja_train.txt', 'w') as f:
    for ja_train in tqdm(ja_trains):
      ja_train = wakati.parse(str(ja_train))
      #print(ja_train)
      f.write(ja_train)
f.close()
