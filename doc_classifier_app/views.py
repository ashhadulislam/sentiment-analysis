from django.shortcuts import render
from django.http import HttpResponse


from django.template import loader




from .scripts import constants
import os

# Create your views here.
def index(request):
	print("hello there")
	print(os.getcwd())
	print(constants.output_results_location)
	print(os.listdir(constants.output_results_location))

	context=constants.update_context()
	template = loader.get_template('doc_classifier_app/index.html')
	return HttpResponse(template.render(context, request))
	# return HttpResponse("Hello, world. You're at the doc_classifier app index.")

from .scripts.train import train_file_model

def train(request):
	print("here in train")
	if request.method == "POST":
		print("In form sub",request.method,request.POST.getlist('models'))
		models_list=request.POST.getlist('models')
		train_file = request.FILES['train_file']
		print("file is ",train_file)
		status=train_file_model(train_file,models_list)
		print("Training status:",status)


		context=constants.update_context()

	template = loader.get_template('doc_classifier_app/index.html')
	return HttpResponse(template.render(context, request))

from .scripts.test import test_model
#to serve files as download
from django.views.static import serve





def test(request):
	print("In form sub")

	if request.method == "POST":
		print("This is post method")
		print("In form sub",request.method,request.POST.getlist('models'))
		models_list=request.POST.getlist('models')
		test_reference_file = request.POST.get("test_reference_file")
		test_text = request.POST.get("text_input")
		print("File to use as reference to test is "+str(test_reference_file)+" and text is "+str(test_text))
		test_file=None

		if 'test_file' in request.FILES:
			test_file = request.FILES['test_file']

		data_result_df,xls_file_path,status=test_model(test_text,test_file,test_reference_file,models_list)
		json_categ=get_categorical_in_json_generic(data_result_df,["SVM","Naive-Bayes"])

		print(json_categ)

		

		sub_key_SVM=list(json_categ["SVM"].keys())
		sub_key_NB=list(json_categ["Naive-Bayes"].keys())

		values_SVM=list(json_categ["SVM"].values())
		values_NB=list(json_categ["Naive-Bayes"].values())

		print(sub_key_SVM,
			sub_key_NB,
			values_NB,
			values_SVM)



		graph_vals = {
		"sub_key_SVM": sub_key_SVM,
		"sub_key_NB": sub_key_NB,
		"values_SVM": values_SVM,
		"values_NB": values_NB
		}

		g = Graph() 
		context = g.get_context_data(graph_vals) 
		return render(request, 'doc_classifier_app/plot_graph.html', context)

		

from wordcloud import WordCloud
import matplotlib
matplotlib.use('TkAgg')

def create_wordcloud(request):
	print("Going to create_wordcloud")

	test_text = request.POST.get("text_input")

	if test_text is not None:
		wordcloud = WordCloud().generate(test_text)

		# Display the generated image:
		# the matplotlib way:
		import matplotlib.pyplot as plt
		plt.imshow(wordcloud, interpolation='bilinear')
		plt.axis("off")

		plt.savefig(constants.word_cloud_image_location+"img.png")




	context={
	"wordcloud":"<img src=/static/img.png>"
	}
	return render(request, 'doc_classifier_app/plot_graph.html', context)






from django.views.generic import TemplateView
import plotly.offline as opy
import plotly.graph_objs as go
import plotly.plotly as py

class Graph(TemplateView):
	template_name = 'graph.html'


	def get_context_data(self, graph_vals,**kwargs):
		context = super(Graph, self).get_context_data(**kwargs)

		
		print("values are")

		'''
{'sub_key_SVM': ['business', 'sport', 'entertainment', 'tech', 'politics'], 
'sub_key_NB': ['business', 'sport', 'entertainment', 'tech', 'politics'], 
'values_SVM': [25, 25, 20, 12, 17], 
'values_NB': [26, 25, 20, 11, 17]}


		'''
		print(graph_vals)

		trace1 = go.Bar(
			x=graph_vals["sub_key_SVM"],
			y=graph_vals["values_SVM"],
			name='SVM'
		)

		trace2 = go.Bar(
			x=graph_vals["sub_key_SVM"],
			y=graph_vals["values_NB"],
			name='Naive-Bayes'
		)

	
		data = [trace1, trace2]



		# layout=go.Layout(title="Meine Daten", xaxis={'title':'x1'}, yaxis={'title':'x2'})

		layout = go.Layout(
			barmode='group'
		)

		figure=go.Figure(data=data,layout=layout)
		div = opy.plot(figure, auto_open=False, output_type='div')

		context['graph'] = div



		labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
		values = [4500,2500,1053,500]

		trace = go.Pie(labels=labels, values=values)

		# py.iplot([trace], filename='basic_pie_chart')

		div2 = opy.plot([trace], auto_open=False, output_type='div')
		print(div2)

		# div2 = opy.plot(fig, auto_open=False, output_type='div')
		context['graph2'] = div

		return context





def get_categorical_in_json_generic(df,list_of_models):
	dict_categories={}
	print(df.head())
	print(list_of_models)
	for model in list_of_models:
		categories=df[model].unique()
		dict_categories[model]={}
		for category in categories:
			count=df[df[model]==category].shape[0]
			dict_categories[model][category]=count
			print(count)
	return dict_categories
		
		


