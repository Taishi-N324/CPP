import MeCab
import ipadic
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import BertJapaneseTokenizer
model = AutoModelForSequenceClassification.from_pretrained('daigo/bert-base-japanese-sentiment')
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

print("文章を入れてください")
example = str(input())
print(nlp(example))
