from django.shortcuts import render
from django.template import loader
from datetime import datetime as dt
from django.http import HttpResponse
import MeCab
import ipadic
import nltk
import socket
import time
import re
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import BertJapaneseTokenizer
from googletrans import Translator
import unicodedata
from sacremoses import MosesTokenizer
import sentencepiece as spm
from fairseq.models.transformer import TransformerModel

mt = MosesTokenizer(lang = 'en')
sp = spm.SentencePieceProcessor(model_file='bpe.model')


import sys

# パスを確認
print(sys.path)


model = TransformerModel.from_pretrained('checkpoints/', checkpoint_file='checkpoint10.pt', data_name_or_path='/Users/Taishi/Desktop/mySite/data_bin')


# Create your views here.

#
# def index_template(request):
#     return render(request, 'index.html')
#
# def index_template_output(request):
#     return render(request, 'output/index.html')


# def input(request):
#     return HttpResponse("Hello world.")





def input(request):
    template = loader.get_template("index.html")
    return HttpResponse(template.render({}, request))




def input_bleu(request):
    template = loader.get_template("index.html")
    return HttpResponse(template.render({}, request))


def output_bleu(request):
    bleu_ref = request.POST["bleu_ref"]
    bleu_text = request.POST["bleu_text"]

    reference = bleu_ref.split()
    hypothesis= bleu_text.split()
    references = [reference]
    list_of_references = [references]
    list_of_hypotheses = [hypothesis]

    result = nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypotheses)
    result = result * 100


    template = loader.get_template("bleu/index.html")

    context = {
        "bleu_ref": bleu_ref,
        "bleu_text": bleu_text,
        "result": result,
    }
    return HttpResponse(template.render(context, request))




def input_google(request):
    template = loader.get_template("index.html")
    return HttpResponse(template.render({}, request))

#
# def input_my_translate(request):
#     template = loader.get_template("index.html")
#     return HttpResponse(template.render({}, request))

# def output_my_translate(request):
#     src = request.POST["google_translate"]
#     tr = Translator(service_urls=['translate.googleapis.com'])
#     trans = tr.translate(src, dest="en").text
#
#     template = loader.get_template("google_translate/index.html")
#     context = {
#         "input_date": trans,
#     }
#     return HttpResponse(template.render(context, request))



#def julius(request)





def output_google(request):
    src = request.POST["google_translate"]
    try:
        tr = Translator(service_urls=['translate.googleapis.com'])
        trans = tr.translate(src, dest="en").text
    except:
        trans ="一日のAPIの呼び出し回数が限度を超えました"

    template = loader.get_template("google_translate/index.html")


    context = {
        "input_date": trans,
        "original_date": src,
    }
    return HttpResponse(template.render(context, request))


def output(request):

    model = AutoModelForSequenceClassification.from_pretrained('daigo/bert-base-japanese-sentiment')
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    input_date_new = nlp(request.POST["date"])
    template = loader.get_template("output/index.html")
    context = {
        "input_date": input_date_new,
    }

    return HttpResponse(template.render(context, request))


def input_julius(request):
    template = loader.get_template("index.html")
    return HttpResponse(template.render({}, request))


def output_julius(request):
    result_julius = []
    src = request.POST["output_julius"]

    # ローカル環境のIPアドレス
    #ここだけ変更を行う。
    host = '192.168.56.1'

    # Juliusとの通信用ポート番号
    port = 10500

    # Juliusにソケット通信で接続
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))
    time.sleep(2)

    # 正規表現で認識された言葉を抽出
    extracted_word = re.compile('WORD="([^"]+)"')
    data = ""

    try:
        while True:
            while (data.find("</RECOGOUT>\n.") == -1):
                data += str(client.recv(1024).decode('shift_jis'))

            # 単語を抽出
            recog_text = ""
            for word in filter(bool, extracted_word.findall(data)):
                recog_text += word

            # 単語を表示
            print("認識結果: " + recog_text)
            result_julius.append(recog_text)
            data = ""
            if len(result_julius) == 5:
                break

    except:
        print('PROCESS END')
        client.send("DIE".encode('shift_jis'))
        client.close()

    template = loader.get_template("output_julius/index.html")

    if len(result_julius)>0:
        context = {'lists': result_julius}
    else:
        context = {'lists': ["音声を認識できませんでした。"]}


    return HttpResponse(template.render(context, request))


def index_test(request):
    context = {'lists': ["データ1", "データ2", "データ3"]}
    return render(request, 'output_test/index.html', context)



def output_test(request):

    template = loader.get_template("output_test/index.html")
    context = {'lists': ["データ1", "データ2", "データ3"]}


    return HttpResponse(template.render(context, request))




def preproc_en(x):
  x = unicodedata.normalize('NFKC', x)
  x = re.sub(mt.AGGRESSIVE_HYPHEN_SPLIT[0], r'\1 - ', x)
  x = mt.tokenize(x, escape = False)
  x = ' '.join(x)
  x = x.lower()
  x = ' '.join(sp.encode(x, out_type = 'str'))
  return x

def translate_fairseq(x):
  x = preproc_en(x)
  print(x,"aaaaaaaaaaaaaaaaaa")

  x = model.translate(x, beam = 5, lenpen = 0.6)
  print(x,"bbbbbbbbbbbbbbbbbb")

  x = ''.join(x.split()).replace('▁', '').strip()
  print(x,"cccccccccc")
  return x


def input_fairseq(request):
    template = loader.get_template("index.html")
    return HttpResponse(template.render({}, request))



def output_fairseq(request):
    x = request.POST["output_fairseq"]



    template = loader.get_template("output_fairseq/index.html")
    x = translate_fairseq(x)
    context = {
        "input_date": x,
    }

    return HttpResponse(template.render(context, request))
