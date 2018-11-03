from django.shortcuts import render
from django.http import HttpResponse


from django.template import loader




from .scripts import constants
import os

# Create your views here.
def index(request):
    print(constants.output_results_location)
    print(os.listdir(constants.output_results_location))

    context=constants.update_context()
    template = loader.get_template('doc_classifier_app/index.html')
    return HttpResponse(template.render(context, request))
    # return HttpResponse("Hello, world. You're at the doc_classifier app index.")

from .scripts.train import train_file_model
def train(request):
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

		data_result_df,status=test_model(test_text,test_file,test_reference_file,models_list)
		


		print("Test status:",status)
		if status==True:
			print("Tested and output received")
			context=constants.update_context(data_result_df)
		elif status==False:
			print("Error:status of test false")
			context=constants.update_context()

	template = loader.get_template('doc_classifier_app/index.html')
	# return HttpResponse("Testing Completed")
	return HttpResponse(template.render(context, request))

