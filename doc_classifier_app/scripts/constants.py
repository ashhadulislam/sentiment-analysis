import os
import pandas as pd
from pathlib import Path



path_to_source=Path(Path(os.getcwd()))

output_results_location=os.path.join(str(path_to_source), "doc_classifier_app/outputs/results/")
vectorlibs_location=os.path.join(str(path_to_source),"doc_classifier_app/outputs/vectorlibs/")
trained_models_location=os.path.join(str(path_to_source),"doc_classifier_app/outputs/trained_models/")
training_files_location=os.path.join(str(path_to_source),"doc_classifier_app/outputs/training_files/")
input_location=os.path.join(str(path_to_source),"doc_classifier_app/inputs/")
word_cloud_image_location=os.path.join(str(path_to_source),"static/")


# output_results_location="doc_classifier_app/outputs/results/"
# vectorlibs_location="doc_classifier_app/outputs/vectorlibs/"
# trained_models_location="doc_classifier_app/outputs/trained_models/"
# training_files_location="doc_classifier_app/outputs/training_files/"
# input_location="doc_classifier_app/inputs/"



class Sentiment_Model:
    def __init__(self, name):
        self.name = name

class Trained_File:
    def __init__(self, name, sentiments):
        self.name = name
        self.sentiments = sentiments
        
class Result:
    def __init__(self, line,sentiment_values):
        self.line = line
        for model_senti_pair in sentiment_values:
            print(model_senti_pair)
            dict_senti={}
            for model,senti in model_senti_pair:
                dict_senti[model]=senti
            self.sentiments.append(dict_senti)



def get_all_sentiments_from_files(training_files_location,file_name_list):
    sentiments_list=[]
    for file_name in file_name_list:
        print(training_files_location+file_name)
        df=pd.read_csv(training_files_location+file_name)
        senti_column=df.columns[2]
        sentiments=df[senti_column].unique()
        sentiments_list.append(sentiments)

    return sentiments_list


def update_context(result_data=None):
    file_name_list=os.listdir(training_files_location)

    #till now we were puttiing the names of these files
    #but now, we are going to put the unique columns in each file as a line
    #example: if file1 contains positive, neutral and negative
    #while file2 contains hate,threat and neutral
    #we shall put the same as the training types

    sentiments_list=get_all_sentiments_from_files(training_files_location,file_name_list)
    print("Trained files available are",file_name_list)


    file_list=[]
    for file_name,sentiments in zip(file_name_list,sentiments_list):
        file_list.append(Trained_File(file_name,sentiments))


    model_list=[Sentiment_Model("SVM"),Sentiment_Model("Naive-Bayes"),Sentiment_Model("Random forest"),Sentiment_Model("NN"),]

    if result_data is None:
        df_dummy_result=pd.read_csv(input_location+"dummy_result.csv")
        
        data_html = df_dummy_result.to_html(index = False)
    else:
        print("Data result received")
        print(result_data.head(5))
        data_html = result_data.to_html(index=False)


      
    
    result_data=data_html
    context_index={
         'model_list':model_list,
        'file_list':file_list,
        'result_list':result_data,
    }

    return context_index