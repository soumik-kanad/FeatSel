import csv
import numpy as np

"""
Usage

>>> from FeatSel import Preprocessor
>>> p=Preprocessor.Preprocessor()
>>> p.load('datasamples/')
>>> your_data=p.Preprocess()

#your_data = {'data':DATA, 'target':TARGET}

"""


class Preprocessor:
	def __init__(self):
		self.data=[];           #list of rows of data
		self.target=[];         #target value
		self.feature_names=[]   #column names
		self.feature_type=[]    #possible types - INT, FLOAT, CATEGORY 

	def load(self, path):
		with open(path+'data.csv', newline='') as csvfile:
			reader = csv.reader(csvfile)
			data_list=list(reader)

		with open(path+'target.csv', newline='') as csvfile:
			reader = csv.reader(csvfile)
			target_list=list(reader)

		#1st line of data.csv is feature_type
		#2nd line of data.csv is feature_name
		#next line onwards data
		## 1st line of target.csv contains text
		## next line onwards it contains the target value of each row 
		self.data=data_list[2:]
		self.target=target_list[1:]
		self.feature_names=data_list[1]
		self.feature_type=data_list[0]

	def isfloat(self,value):
		try:
			float(value)
			return True
		except ValueError:
			return False

	def isint(self,value):
		try:
			int(value)
			return True
		except ValueError:
			return False

	def fill_empty_with_avg(self):
		for i in range(len(self.feature_names)):
			if self.feature_type[i] == 'INT':
				#find row average
				temp_list=[]
				for row in self.data:
					if self.isint(row[i]):
						temp_list.append(row[i])
				average=np.floor(np.mean(temp_list))
				#replace unknown values by average
				for row in self.data:
					if not(self.isint(row[i])):
						row[i]=average

			elif self.feature_type[i] == 'FLOAT':
				#find row average
				temp_list=[]
				for row in self.data:
					if self.isfloat(row[i]):
						temp_list.append(row[i])
				average=np.mean(temp_list)
				#replace unknown values by average
				for row in self.data:
					if not(self.isfloat(row[i])):
						row[i]=average

			else:
				pass



	def normalize(self, column_no):
		#to be done
		pass

	def Preprocess(self):
		#for every column with id i
		for i in range(len(self.feature_names)):
			if self.feature_type[i] == 'INT':
				#convert to integet
				for row in self.data:
					if self.isint(row[i]):
						row[i]=int(row[i])
				self.normalize(i)
			elif self.feature_type[i] == 'FLOAT':
				#convert to float
				for row in self.data:
					if self.isfloat(row[i]):
						row[i]=float(row[i])
				self.normalize(i)
			else:
				categories=list(set([row[i] for row in self.data]))
				category_value={}
				for j in  range(len(categories)):
					category_value[categories[j]]=j

				for row in self.data:
					row[i]=category_value[row[i]]
		self.fill_empty_with_avg()
		return {'data':self.data , 'target':self.target}



#path='../datasamples/'
