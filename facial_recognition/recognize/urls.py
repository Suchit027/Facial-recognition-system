# recognize/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register_person, name='register'),
    path("identify/", views.identify_person, name='identify'),
]
