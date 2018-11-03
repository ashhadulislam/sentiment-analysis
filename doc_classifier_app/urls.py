from django.urls import path

from . import views



app_name = 'doc_classifier_app'

urlpatterns = [
	# ex: /senti/
    path('', views.index, name='index'),


    path('train/', views.train, name='train'),

	path('test/', views.test, name='test'),
]