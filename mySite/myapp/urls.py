from django.urls import path
from . import views

urlpatterns = [
    #path('templates/', views.index_template, name='index_template'),
    #path('templates/', views.index_template, name='index_template'),
    path('templates/', views.input, name='input'),
    path('templates/output', views.output,name='output'),
    path('templates/', views.input_google, name='input_google'),
    path('templates/google_translate', views.output_google,name='output_google'),
    path('templates/bleu', views.output_bleu,name='output_bleu'),
    path('templates/output_test', views.output_test,name='output_test'),
    path('templates/output_julius', views.output_julius,name='output_julius'),
    path('templates/', views.input_julius,name='input_julius')
    
    #path('templates/my_translate', views.input_my_translate,name='input_my_translate'),
    #path('templates/my_translate', views.output_my_translate,name='output_my_translate')
]
