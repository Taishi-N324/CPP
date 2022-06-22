#インストール
#!pip install mecab-python3

#辞書インストール
#!pip install unidic-lite

#! git clone https://github.com/Taishi-N324/CPP

import nltk

#nltk.download()


import MeCab
import csv
import re
from tqdm import tqdm
import os

new_dir_path = 'data/'

try:
    os.mkdir(new_dir_path)
except:
    pass


en_trains = []
ja_trains = []

tagger = MeCab.Tagger()
wakati = MeCab.Tagger("-Owakati")
'''
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

test = []

last = [')','%','?','!',';',']','(', ':', '.',"'"]

with open('data/en_train.txt', 'w') as f:
    for en_train in tqdm(en_trains):
        test.append(en_train[-1])

        if en_train[-1] in last:
            en_train_last = en_train[-1]
            en_train = en_train[:-1]+ " " + en_train_last
        f.write(en_train)
        f.write('\n')

f.close()
'''

# with open('data/ja_train.txt', 'w') as f:
#     for ja_train in tqdm(ja_trains):
#       ja_train = wakati.parse(str(ja_train))
#       #print(ja_train)
#       f.write(ja_train)
# f.close()

remove_list = ["^","© ","^ ","©"]

with open('/Users/Taishi/Desktop/mySite/TokyoTech_en.txt') as f:
    for s_line in f:
        text = ""
        words = nltk.word_tokenize(s_line)
        for word in words:
            word = word.replace("^","")
            word = word.replace("©","")
            text += word + " "
        en_trains.append(text)

with open('/Users/Taishi/Desktop/mySite/TokyoTech_ja.txt') as f:
    for s_line in f:
      texts = [wakati.parse(s_line)]
      start = ""
      for text in texts:
          text = text.replace("^","")
          text = text.replace("©","")
          start += text + " "
      ja_trains.append(start)

with open('data/TokyoTech_en.txt', 'w') as f:
    for en_train in tqdm(en_trains):
        #print(ja_train)
        if en_train != "":
            f.write(en_train)
            f.write("\n")
f.close()

with open('data/TokyoTech_ja.txt', 'w') as f:
    for ja_train in tqdm(ja_trains):
        ja_train = wakati.parse(str(ja_train))
        judge = [ja_train] == ['\n']
        if judge == False:
            f.write(ja_train)

        #print([ja_train])
        #print(ja_train)

        #f.write(ja_train)


f.close()

#print(ja_trains)
# print(en_trains)
