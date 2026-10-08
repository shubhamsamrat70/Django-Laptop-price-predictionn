from django.urls import path
from .import views

urlpatterns = [
    path('housePredict', views.housePredict, name='housePredict'),
]