from django.contrib import admin
from django.urls import path
from article import views

urlpatterns = [
    path('', views.home, name = "home"),
    path('article/', views.article, name = "article"),
    path('articledownload/', views.articledownload, name = "articledownload"),
    path('transcript/', views.transcript, name = "transcript"),
    path('transcriptdownload/', views.transcriptdownload, name = 'transcriptdownload'),
    path('contactus/', views.contactus, name = 'contactus'),
    path('thankyou/', views.thankyou, name = 'thankyou'),
    path('document/', views.document, name = 'document'),
    path('documentdownload/', views.documentdownload, name = 'documentdownload'),
    path('accuracy/', views.accuracy, name = 'accuracy'),
]