from django.urls import path

from . import views

app_name = 'Summarizer'

urlpatterns = [
    path('', views.index, name='index'),
    path('jake', views.jake_view, name='jake'),
    path('jheymie', views.jheymie_view, name='jheymie'),
    path('renwell', views.renwell_view, name='renwell'),
    path('cyrus', views.cyrus_view, name='cyrus')
]
