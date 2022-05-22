from django.shortcuts import render
from django.template import loader
from datetime import datetime as dt
from django.http import HttpResponse


import MeCab
import ipadic
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import BertJapaneseTokenizer
from googletrans import Translator

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



def input_google(request):
    template = loader.get_template("index.html")
    return HttpResponse(template.render({}, request))

def output_google(request):
    src = request.POST["google_translate"]
    tr = Translator(service_urls=['translate.googleapis.com'])
    trans = tr.translate(src, dest="en").text

    template = loader.get_template("google_translate/index.html")
    context = {
        "input_date": trans,
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
