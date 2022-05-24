from django.urls import path
from . import views

urlpatterns = [
    #path('templates/', views.index_template, name='index_template'),
    #path('templates/', views.index_template, name='index_template'),
    path('templates/', views.input, name='input'),
    path('templates/output', views.output,name='output'),
    path('templates/', views.input_google, name='input_google'),
    path('templates/google_translate', views.output_google,name='output_google')
    #path('templates/my_translate', views.input_my_translate,name='input_my_translate'),
    #path('templates/my_translate', views.output_my_translate,name='output_my_translate')
]
