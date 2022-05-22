from django.shortcuts import render

# Create your views here.

 
def index_template(request):
    return render(request, 'index.html')

from django.http import HttpResponse

def input(request):
    return HttpResponse("Hello world.")
    
    
from django.http import HttpResponse
from django.template import loader
from datetime import datetime as dt

def input(request):
    template = loader.get_template("templates/index.html")
    return HttpResponse(template.render({}, request))

def output(request):
    #weekdays = ["月","火","水","木","金","土","日"]
    input_date = request.POST["date"]
    #date_value = dt.strptime(input_date, "%Y-%m-%d")
    #weekday_value = weekdays[date_value.weekday()]

    template = loader.get_template("templates/output/index.html")
    context = {
        "input_date": input_date,
        #"weekday": weekday_value,
    }
    return HttpResponse(template.render(context, request))
