import pandas as pd

from . import preprocess
# from preprocess import PreProcess

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

from . import constants

def pre_process_data(df):

	#the pre processing part
	column_name = df.columns[0]
	data=df
	pre_processor = preprocess.PreProcess(data, column_name)	

	data = pre_processor.clean_html()
	data = pre_processor.remove_non_ascii()
	data = pre_processor.remove_spaces()
	data = pre_processor.remove_punctuation()
	data = pre_processor.stemming()
	data = pre_processor.lemmatization()
	data = pre_processor.stop_words()

	return data


def get_train_vectors(data,identifier):
	col1=data.columns[0]
	col2=data.columns[1]

	# train_x, test_x, train_y, test_y = train_test_split(data[col1], data[col2], test_size=0.00)
	# print(train_x.shape, train_y.shape)
	# print(test_x.shape, test_y.shape)
	tfidf_transformer = TfidfVectorizer(min_df=1)
	train_vectors = tfidf_transformer.fit_transform(data[col1])
	joblib.dump(tfidf_transformer, str(constants.vectorlibs_location)+str(identifier)+'_vectorizer.pkl')
	



	return train_vectors




#this function is the entry point for training
def train_file_model(filename, models_list):

	print("constant = ",constants.vectorlibs_location,constants.trained_models_location)

	print(models_list)
	print("In train file model, going to read file",filename)	

	#this identifier will be used to save the pkl files
	identifier=filename


	df=pd.read_csv(filename)
	print(df.head())

	data=pre_process_data(df)

	
	

	###############################################################################
	# Feature extraction
	###############################################################################
	
	train_vectors=get_train_vectors(data,identifier)

	print("After vect")
	print(data.head())

	col1=data.columns[0]
	col2=data.columns[1]



	 ###############################################################################
	# Perform classification with SVM, kernel=linear
	for each_model in models_list:
		print(each_model)
		if each_model == "SVM":
			model = svm.SVC(kernel='linear')
			model.fit(train_vectors, df[col2])
			

		elif each_model=="Naive-Bayes":
			model = MultinomialNB()
			print("going to naive baye")
			model.fit(train_vectors, df[col2])
		else:
			return False

		print("Saving training file")
		df.to_csv(constants.training_files_location+str(filename))
		print("Train file saved")
		print("going to store in ",str(constants.trained_models_location)+str(filename)+"_"+str(each_model)+'.pkl')	
		joblib.dump(model, str(constants.trained_models_location)+str(filename)+"_"+str(each_model)+'.pkl')


	return True
