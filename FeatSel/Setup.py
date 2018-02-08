from os.path import dirname, join, exists, isdir
import argparse
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from FeatSel import Data

"""
Setup is used to set the parameters for the DatasetWrapper.

Call setup with atleast these parameters:

    #Your data in this format
    your_data = {'data':DATA, 'target':TARGET}

    def your_evaluator_function(data, subset):
        #subset is a list of feature indexes corresponding to your_data.data.
        #compute error for the subset
        return error

    customSetup = Setup(data=your_data, evaluatorFunction=your_evaluator_function)

Args:
    data : A dictionary containg your_data
    evaluatorFunction : A lambda function to your_evaluator_function.

Returns:
    type: Setup object

"""

class Setup:
    def __init__(self, kwargs=None):
        if kwargs != None:
            for key, value in kwargs.items():
                if key=='data' and isinstance(value, dict):
                    value = Data(value)
                setattr(self, key, value)
        else:
            self.loadDefaults()
        self.verifySetup()

    def loadDefaults(self):
        self.data_sample_folder = join(dirname(dirname(__file__)), 'datasamples')
        self.test_size = 0.4
        self.data = datasets.load_boston() #TODO: change data to be a dataset loaded from samlpe folder
        self.f_size = len(self.data.data[0])

        # Evaluate subset of features
        def _DefaultEvaluate(data, feature_subset):
            X_train, X_test, Y_train, Y_test, = train_test_split(data.data[:,feature_subset], data.target, test_size=self.test_size)
            model = LinearRegression().fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            return mean_squared_error(Y_test, Y_pred)

        self.evaluatorFunction = _DefaultEvaluate

    def verifySetup(self):
        pass
