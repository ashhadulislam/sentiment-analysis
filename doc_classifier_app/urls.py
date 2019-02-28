from django.urls import path

from . import views



app_name = 'doc_classifier_app'

urlpatterns = [
	# ex: /senti/
    path('', views.index, name='index'),




    path('train/', views.train, name='train'),

	path('test/', views.test, name='test'),

	path('show_wc/', views.create_wordcloud, name='create_wordcloud'),


	path('get_result_sentiment/', views.get_result_sentiment, name='get_result_sentiment'),
]