import csv
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os



class Preprocessor(object):
	"""
    An instance of this class helps in loading and preprocessing the data
	
	Members:
        type : Type of preprocessing. Only 'Default' implemented
        data : Contains data without the target as lists of lists
        target : Contains target of each row
        feature_names : Name of columns stored in list of strings
        class_name : 

	Usage:
	>>> p=Preprocessor().load('datasamples/','winequality.csv').Preprocess()

    """

	def __init__(self, type='Default'):
		self.type=type
		self.data=[];           #list of rows of data
		self.target=[];         #target value
		self.feature_names=[]   #column names
		self.class_name=""


	def load(self, path, filename):
		"""
        Loads data from csv, json and xlsx files

        Args :
            path : Path to the directory containing the folder
            filename : Name of the file

        Returns:
            self : returns a Preprocessor object with feature_names, class_name, data, target loaded from the file
        """

		absolute_path = os.path.join(path, filename)
		assert os.path.exists(absolute_path)
		if filename.lower().endswith('.csv'):
			csv_load = pd.read_csv(absolute_path)	
			self.feature_names = csv_load.columns[:-1]
			self.class_name = csv_load.columns[-1]
			self.data = csv_load[self.feature_names]
			self.target = csv_load[self.class_name]
		elif filename.lower().endswith('.json'):
			# TODO: NOt tested, please test.
			json_load = pd.read_json(absolute_path)
			self.feature_names = json_load.columns[:-1]
			self.class_name = json_load.columns[-1]
			self.data = json_load[self.feature_names]
			self.target = json_load[self.class_name]
			pass
		elif filename.lower().endswith('.xlsx'):
			xlsx_load = pd.read_excel(absolute_path)
			self.feature_names = xlsx_load.columns[:-1]
			self.class_name = xlsx_load.columns[-1]
			self.data = xlsx_load[self.feature_names]
			self.target = xlsx_load[self.class_name]

		return self

	def Preprocess(self):
		"""
        Preprocesses the loaded data

        Args :

        Returns:
            formated_data : Returns a dictionary with keys 'data', 'target', 'feature_names', 'class_names' which are the same as the data elements of the Preprocessor object.
        """

		if self.type == 'Default':

			#converting string type columns into category type
			categorical_columns=[]
			for col in self.data.columns:
				if self.data[col].dtype==np.object:  #by default string data was of object type
					categorical_columns.append(col)
					self.data[col] = self.data[col].astype('category')
			

			#for one hot encoding
			self.data=pd.get_dummies(self.data,columns=categorical_columns) 


			#transformation and scaling
			formated_data = {}
			
			# print(self.data['alcohol'])

			imputer = preprocessing.Imputer().fit(self.data)
			formated_data['data'] = imputer.transform(self.data)

			# print(self.data.columns, formated_data['data'][-1])

			# For handling missing values
			scaler = preprocessing.StandardScaler()
			formated_data['data'] = scaler.fit_transform(formated_data['data'])

			# print(formated_data['data'][:,:-1])

			formated_data['target'] = self.target
			#formated_data['feature_names'] = self.feature_names
			formated_data['feature_names'] = self.data.columns
			formated_data['class_name'] = self.class_name
			return formated_data
		else:
			raise NotImplementedError



