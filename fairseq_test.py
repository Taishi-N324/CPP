import re
import unicodedata
from sacremoses import MosesTokenizer
import sentencepiece as spm
from fairseq.models.transformer import TransformerModel


mt = MosesTokenizer(lang = 'en')
sp = spm.SentencePieceProcessor(model_file='bpe.model')
#print("1111111111111111111111111111111111111111111111111111111")
#model = TransformerModel.from_pretrained('checkpoints/', checkpoint_file='checkpoint10.pt', data_name_or_path='/Users/Taishi/Desktop/mySite/data_bin')
model = TransformerModel.from_pretrained('checkpoints/', checkpoint_file='checkpoint_best_tokyotech.pt', data_name_or_path='/Users/Taishi/Desktop/mySite/data_bin_tokyotech')




def preproc_en(x):
  x = unicodedata.normalize('NFKC', x)
  x = re.sub(mt.AGGRESSIVE_HYPHEN_SPLIT[0], r'\1 - ', x)
  x = mt.tokenize(x, escape = False)
  x = ' '.join(x)
  x = x.lower()
  x = ' '.join(sp.encode(x, out_type = 'str'))
  return x

def translate(x):
  x = preproc_en(x)
  #print("aaaaa")
  x = model.translate(x, beam = 5, lenpen = 0.6)
  #print("bbbbbbbb")
  x = ''.join(x.split()).replace('▁', '').strip()
  return x

while True:
    x = input('Please enter English > ')
    if not x:
        break
    x = translate(x)
    print('日本訳 > ' + x)
