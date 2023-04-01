import torch
import configparser
from sklearn.linear_model import Ridge, lars_path
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import torch.autograd as autograd
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import torchvision.models as tvmodels
from datetime import datetime
from functools import partial 
from sklearn.utils import check_random_state
import sklearn.metrics 
import os
import json
import string 

def currentDir():
    return os.path.dirname(os.path.realpath(__file__))
    
def parentDir(mydir):
    return str(Path(mydir).parent.absolute())
    
def readConfigFile(configfile):
    parameters = {}
    # read parameters from config file
    config = configparser.ConfigParser()
    config.read(configfile)

    p_default = config['DEFAULT']

    parameters['regressiveDataPath'] = p_default['RegressiveDataFullPath'] 
    parameters['regressiveInversePath'] = p_default['RegressiveInverseFullPath']
    parameters['explainedModelPath'] = p_default['ExplainedModelFullPath'] 
    parameters['originalTrainDataPath'] = p_default['OriginalTrainDataFullPath'] 
    parameters['nClasses'] = p_default.getint('ClassNum')
    parameters['logpath'] = p_default['LogFullPath'] 
    return parameters
    
class ExperimentDataset(Dataset):

    def __init__(self, datafilepath):
        full_data_table = np.genfromtxt(datafilepath, delimiter=',')
        data = torch.from_numpy(full_data_table).float()
        self.samples = data[:, :-1]
        
        self.labels = data[:, -1]
        min, _ = self.samples.min(dim=0)
        max, _ = self.samples.max(dim=0)
        self.feature_min = min
        self.feature_max = max

        self.samples = (self.samples - self.feature_min)/(self.feature_max-self.feature_min)
        self.mean_attr = self.samples.mean(dim=0)
                      
    def getNumpyData(self):
        return self.samples.numpy()
                
    def getNumpyLabels(self):
        return self.labels.numpy()
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]
    
def readPrivateTrainingData(datafilepath, trainingPortion=0.8, batchSize=64, binaryClassification=False):    
    expset = ExperimentDataset(datafilepath)
    print("Creating dataset, len(samples): {}.".format(len(expset)) )
    if binaryClassification:
        print("Positive labels percentage: {}".format((expset.labels > 0).sum() / float(len(expset))))

    train_len = int(len(expset) * trainingPortion)
    test_len = len(expset) - train_len
    trainset, testset= torch.utils.data.random_split(expset, [train_len, test_len])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=True)
    print("len(trainloader):", len(trainloader))
    print("len(testloader):", len(testloader))
    x, y = expset[0]
    print("Sample x:", x)
    print("len(x):", len(x))
    print("Sample y:", y)
    return trainset, trainloader, testset, testloader, expset

class GlobalPreModel_LR(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super(GlobalPreModel_LR, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=True),
            nn.Sigmoid(), 
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.dense(x)
            
def initialFeatureClassNames():
    feature_names = ["age", 
    "job", 
    "marital", 
    "education", 
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"]

    class_names = ["NoDeposit", "YesDeposit"]
    return feature_names, class_names
    
def getKernel(kernel_width):
    def kernel(d, kernel_width):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
    kernel_fn = partial(kernel, kernel_width=kernel_width) # fix the value of kernel_width
    return kernel_fn
    
def __data_inverse(feature_num, num_samples):
    """Generates a neighborhood around a prediction."""
    sample_around_instance = False
    categorical_features = range(feature_num)

    random_state = check_random_state(manualseed)
    data = random_state.normal(0, 1, num_samples * feature_num).reshape(num_samples, feature_num) # Normal(0,1)
    if sample_around_instance:
        pass
    else:
        data = data 
    # first_row = data_row
    # data[0] = data_row.copy()
    inverse = data.copy()
    for column in categorical_features:
        # process inverse
        pass
    # inverse[0] = data_row
    return data, inverse
    
def compNeighborDistances(neighborData, datarow, distance_metric):
    # Compute the distance matrix from a vector array X and optional Y
    # Return a 1-D flattened array
    return sklearn.metrics.pairwise_distances(neighborData, datarow, metric=distance_metric).ravel()

def compNeighborLabels(trainedModel, neighborData):
    yss = trainedModel(torch.from_numpy(neighborData).float()).detach().numpy()
    return yss
    
def convert_and_round(values):
    return ['%.2f' % v for v in values]

class Explanation(object):
    """Object returned by explainers."""

    def __init__(self, feature_num, class_num, feature_names=None, class_names=None):
        """ 
        Args:
            feature_names: list of feature names
            class_names: list of class names (only used for classification)
        """ 
        
        self.local_exp = {}
        self.intercept = {}
        self.score = None
        self.local_pred = None
        self.scaled_data = None
        self.class_names = class_names
        self.feature_names = feature_names
        if class_names is None:
            self.class_names = [str(x) for x in range(class_num)]
        if feature_names is None:
            self.feature_names = [str(x) for x in range(feature_num)]

        self.predict_proba = None    # the prediction for the instance being explained
        
    def map_exp_ids(self, exp):
        """Maps ids to feature names.
        Args:
            exp: list of tuples [(id, weight), (id,weight)]
        Returns:
            list of tuples (feature_name, weight)
        """
        return [(self.feature_names[x[0]], x[1]) for x in exp]

    def available_labels(self):
        """
            Returns the list of classification labels for which we have any explanations.
        """
        ans = self.local_exp.keys()
        return list(ans)

    def as_list(self, label=1, **kwargs):
        """Returns the explanation as a list.
        Args:
            label: desired label. If you ask for a label for which an
                explanation wasn't computed, will throw an exception.
                Will be ignored for regression explanations.
            kwargs: keyword arguments, passed to domain_mapper
        Returns:
            list of tuples (representation, weight), where representation is
            given by domain_mapper. Weight is a float.
        """
        label_to_use = label
        ans = self.map_exp_ids(self.local_exp[label_to_use], **kwargs)

        return ans

    def as_map(self):
        """Returns the map of explanations.
        Returns:
            Map from label to list of tuples (feature_id, weight).
        """
        return self.local_exp

   
  
class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self, kernel_fn, verbose=False):
        """ Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose

    def explain_instance_with_data(self, neighborhood_data, neighborhood_labels, distances, label):
        """Takes perturbed data, labels and distances, returns explanation.
        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation

        Returns:
            (intercept, exp, score):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
        """

        weights = self.kernel_fn(distances).squeeze()
        labels_column = neighborhood_labels[:, label]
        used_features = np.array(range(data.shape[1]))
        
        ridge = RidgeSGD(alpha=1, fit_intercept=True)
        ridge.ridgeSGDFit(neighborhood_data[:, used_features], labels_column, sample_weight=weights)

        if self.verbose:
            ridge.printFittedModel(len(used_features))
        return ridge.coef_, ridge.intercept_, ridge.n_iter_
  
class RidgeSGD(object):
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-3):
        self.alpha=alpha
        self.fit_intercept=fit_intercept
        self.max_iter=max_iter
        self.tol=tol
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        
    def ridgeSGDFit(self, X, y, sample_weight=None):        
        self.coef_, self.intercept_, self.n_iter_ = self.sag_solver(X, y, self.fit_intercept, sample_weight, self.alpha, max_iter=self.max_iter, tol=self.tol)
        return self.coef_, self.intercept_, self.n_iter_

    def printFittedModel(self, n_features):
        for i in range(n_features):
            print("coef[{}]: {}".format(i, self.coef_[i]))
        print("intercept:", self.intercept_)
        print("n_iter:", self.n_iter_)
   
    def sag_solver(self, X, y, fit_intercept=True, sample_weight=None, alpha=1., max_iter=1000, tol=0.001):
        def fmax(a, b):
            return a if a > b else b
        n_samples, n_features = X.shape
        # As in SGD, the alpha is scaled by n_samples.
        alpha_scaled = float(alpha) / n_samples 
        n_classes = 1 
        coef_init =  np.zeros((n_features + int(fit_intercept), 1)) # coef: array(n_features+n_intercept, 1)
        if fit_intercept:
            intercept_init = coef_init[-1, :]
            coef_init = coef_init[:-1, :]
        else:
            intercept_init = np.zeros(n_classes)

        weights = np.zeros(n_features)
        previous_weights = np.zeros(n_features)
        sum_gradient = np.zeros(n_features)
        gradient_memory = np.zeros((n_samples, n_features))
        intercept = 0.0
        intercept_sum_gradient = 0.0 
        intercept_decay = 1.0
        intercept_gradient_memory = np.zeros(n_samples)
        seen = set()
        rng = np.random.RandomState(77)
        max_squared_sum = 20    # to be defined
        L = max_squared_sum + int(fit_intercept) + alpha_scaled
        step_size = 1. / L
        n_iter = 0 

        for epoch in range(max_iter):
            for k in range(n_samples):
                idx = int(rng.rand(1) * n_samples)
                entry = X[idx]
                seen.add(idx)
                p = np.dot(entry, weights) + intercept # inner product of 1-D arrays
                gradient = p - y[idx]
                if sample_weight is not None:
                    gradient = gradient * sample_weight[idx]
                update = entry * gradient + alpha_scaled * weights   
                gradient_correction = update - gradient_memory[idx]
                sum_gradient += gradient_correction
                gradient_memory[idx] = update

                if fit_intercept:
                    gradient_correction = (gradient - intercept_gradient_memory[idx])
                    intercept_gradient_memory[idx] = gradient
                    intercept_sum_gradient += gradient_correction

                    intercept = intercept - (step_size * intercept_sum_gradient / len(seen) * intercept_decay)

                weights = weights - step_size * sum_gradient / len(seen)
            # The iterations will stop when max(change in weights) / max(weights) < tol.
            max_change = 0.0
            max_weight = 0.0
            for i in range(n_features):
                max_weight = fmax(max_weight, abs(weights[i]))
                max_change = fmax(max_change, abs(weights[i] - previous_weights[i]))
                previous_weights[i] = weights[i]
            if (max_weight != 0 and max_change/max_weight <= tol) or (max_weight == 0 and max_change ==0):
                n_iter = epoch + 1 
                break

        return weights, intercept, n_iter 
  
def resetRandomStates(manualseed=47):
    random.seed(manualseed)
    torch.manual_seed(manualseed)
    np.random.seed(manualseed)
    
def readOriginalTrainData(dataPath):
    _, _, _, _, entireDataset = readPrivateTrainingData(dataPath, binaryClassification=True)
    training_data = entireDataset.getNumpyData()
    training_labels = entireDataset.getNumpyLabels()
    print("training_data.shape: {}; training_data.shape: {}".format(training_data.shape, training_labels.shape) )
    return training_data, training_labels

def loadTrainedModel(modelPath, inDim, outDim, modelType="LR"):
    model = None 
    if modelType == "LR":
        # load trained LR model 
        model = GlobalPreModel_LR(inDim, outDim) 
        model.load_state_dict(torch.load(modelPath))
    return model 

def loadRegressiveData(dataPath, inversePath):
    # neighboring data 
    # data for computing distances, inverse for computing model predictions

    with open(dataPath, 'rb') as f:
        data = np.load(f)
    with open(inversePath, 'rb') as f: 
        inverse = np.load(f)
    # data, inverse = __data_inverse(feature_num, num_samples=10000)
    print("Shape of the neighboring dataset:", data.shape)
    return data, inverse 

if __name__ == "__main__" :
    # initialize some parameters
    resetRandomStates(manualseed=47)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create Lime parameters
    feature_names, class_names = initialFeatureClassNames()
    
    parameters = readConfigFile("/home/xinjian/Desktop/MP-SPDZ/Programs/Source/lime-baseline-files/lime-config.ini") 
    
    # training dataset
    training_data, _ = readOriginalTrainData(parameters['originalTrainDataPath'])

    # the instance to be explained
    data_row = training_data[0]  
    feature_num = data_row.shape[0]
    class_num = parameters['nClasses']
    # print("Class_num", class_num)
    
    lrmodel = loadTrainedModel(parameters['explainedModelPath'], inDim=feature_num, outDim=class_num, modelType="LR")
        
    # kernel
    kernel_fn = getKernel(np.sqrt(training_data.shape[1]) * .75)
    
    data, inverse = loadRegressiveData(dataPath=parameters['regressiveDataPath'], inversePath=parameters['regressiveInversePath'])
    
    distances = compNeighborDistances(data, data_row.reshape(1, -1), distance_metric='euclidean')

    yss = compNeighborLabels(lrmodel, inverse)
    
    if not np.allclose(yss.sum(axis=1), 1.0):
        # Returns True if two arrays are element-wise equal within a tolerance
        print(""" Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions). """)     

    # values = convert_and_round(data_row)
    
    ret_exp = Explanation(feature_num, class_num, feature_names, class_names)
    
    base = LimeBase(kernel_fn, verbose=True)    
    
    label = 0
    base.explain_instance_with_data(data, yss, distances, label) 

    print(" ------------------------------ FINISHED ------------------------------ ")
    
