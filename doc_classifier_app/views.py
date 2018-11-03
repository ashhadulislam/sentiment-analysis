from django.shortcuts import render
from django.http import HttpResponse


from django.template import loader


from . import constants

import os

# Create your views here.
def index(request):
    print(constants.output_results_location)
    print(os.listdir(constants.output_results_location))

    context=constants.update_context()
    template = loader.get_template('doc_classifier_app/index.html')
    return HttpResponse(template.render(context, request))
    # return HttpResponse("Hello, world. You're at the doc_classifier app index.")
