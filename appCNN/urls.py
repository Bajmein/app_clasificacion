from django.contrib import admin
from django.urls import path
from app_red import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.predict, name='predict'),
    path('upload/', views.upload, name='upload'),
]
