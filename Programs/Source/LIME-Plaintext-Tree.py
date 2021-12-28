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
import numbers
import sys
print("sys.getrecursionlimit():", sys.getrecursionlimit())
sys.setrecursionlimit(5000)


SIZE_MAX = 1e8
_TREE_UNDEFINED = -2
_TREE_LEAF = -1

EPSILON = 1e-6
IS_FIRST = 1
IS_NOT_FIRST = 0
IS_LEFT = 1
IS_NOT_LEFT = 0
INFINITY = 1e8


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

        self.samples = (self.samples - self.feature_min) / (self.feature_max - self.feature_min)
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
    print("Creating dataset, len(samples): {}.".format(len(expset)))
    if binaryClassification:
        print("Positive labels percentage: {}".format((expset.labels > 0).sum() / float(len(expset))))

    train_len = int(len(expset) * trainingPortion)
    test_len = len(expset) - train_len
    trainset, testset = torch.utils.data.random_split(expset, [train_len, test_len])
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

    kernel_fn = partial(kernel, kernel_width=kernel_width)  # fix the value of kernel_width
    return kernel_fn


def __data_inverse(feature_num, num_samples):
    """Generates a neighborhood around a prediction."""
    sample_around_instance = False
    categorical_features = range(feature_num)

    random_state = check_random_state(manualseed)
    data = random_state.normal(0, 1, num_samples * feature_num).reshape(num_samples, feature_num)  # Normal(0,1)
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


class MSERegressionCriterion():
    """The computation takes linear time with `n_samples`
    by using ::
        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
    """

    def __init__(self, n_outputs, n_samples):
        """
        n_outputs : The number of targets to be predicted
        n_samples : The total number of samples to fit on
        """
        self.n_outputs = n_outputs
        self.n_samples = n_samples

    def init(self, y, sample_weight, weighted_n_samples, samples, start, end):
        """This initializes the criterion at node samples[start:end] and children
        samples[start:start] and samples[start:end].
        Parameters
        ----------
        y             :  y is a buffer that can store values for n_outputs target variables
        sample_weight :  The weight of each sample
        weighted_n_samples :  The total weight of the samples being considered
        samples       :  Indices of the samples in X and y, where samples[start:end]
            correspond to the samples in this node
        start         :  The first sample to be used on this node
        end           :  The last sample used on this node

        """
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        w = 1.0

        self.sq_sum_total = 0.0
        self.sum_total = np.zeros((self.n_outputs))
        self.sum_left = np.zeros((self.n_outputs))
        self.sum_right = np.zeros((self.n_outputs))

        for p in range(start, end):
            i = samples[p]  # the index of current sample

            if sample_weight is not None:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]  # the target output for current sample
                w_y_ik = w * y_ik
                self.sum_total[k] += w_y_ik  # sum(w * y)
                self.sq_sum_total += w_y_ik * y_ik  # sum(w * y**2)

            self.weighted_n_node_samples += w  # sum(w)

        # Reset to pos=start
        self.reset()
        return 0

    def reset(self):
        """Reset the criterion at pos=start."""
        for i in range(self.n_outputs):
            self.sum_left[i] = 0.0
            self.sum_right[i] = self.sum_total[i]

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        return 0

    def reverse_reset(self):
        """Reset the criterion at pos=end."""
        for i in range(self.n_outputs):
            self.sum_right[i] = 0.0
            self.sum_left[i] = self.sum_total[i]

        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end
        return 0

    def update(self, new_pos):
        """Updated statistics by moving samples[pos:new_pos] to the left child.
        This updates the collected statistics by moving samples[pos:new_pos]
        from the right child to the left child.
        Parameters
        ----------
        new_pos : New starting index position of the samples in the right child
        """
        sum_left = self.sum_left
        sum_right = self.sum_right
        sum_total = self.sum_total

        sample_weight = self.sample_weight
        samples = self.samples

        pos = self.pos
        end = self.end
        w = 1.0

        # Update statistics up to new_pos
        # Given that sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]  # sample index

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    sum_left[k] += w * self.y[i, k]

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    sum_left[k] -= w * self.y[i, k]

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples - self.weighted_n_left)
        for k in range(self.n_outputs):
            sum_right[k] = sum_total[k] - sum_left[k]

        self.pos = new_pos
        return 0

    def node_value(self, dest):
        """Compute the node value of samples[start:end] and save the value into dest.
        Parameters
        ----------
        dest : The memory address where the node value should be stored.
        """

        # sum_total ->  sum(w * y)
        # weighted_n_node_samples -> sum(w)
        for k in range(self.n_outputs):
            dest[k] = self.sum_total[k] / self.weighted_n_node_samples

    def node_impurity(self):
        """Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of samples[start:end]. The smaller the impurity the better.
        """
        sum_total = self.sum_total
        # sum_total ->  sum(w * y)
        # sq_sum_total -> sum(w * y**2)
        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_outputs):
            impurity -= (sum_total[k] / self.weighted_n_node_samples) ** 2.0  # y**2 -y\bar**2, like variance

        return impurity / self.n_outputs

    def proxy_impurity_improvement(self):
        """Compute a proxy of the impurity reduction.
        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.
        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        sum_left = self.sum_left
        sum_right = self.sum_right

        proxy_impurity_left = 0.0
        proxy_impurity_right = 0.0

        for k in range(self.n_outputs):
            proxy_impurity_left += sum_left[k] * sum_left[k]
            proxy_impurity_right += sum_right[k] * sum_right[k]

        return (proxy_impurity_left / self.weighted_n_left + proxy_impurity_right / self.weighted_n_right)

    def impurity_improvement(self, impurity_parent, impurity_left, impurity_right):
        """Compute improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child,
        Parameters
        ----------
        impurity_parent : The initial impurity of the parent node before the split
        impurity_left : The impurity of the left child
        impurity_right : The impurity of the right child
        Return
        ------
        double : improvement in impurity after the split occurs
        """
        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity_parent - (self.weighted_n_right / self.weighted_n_node_samples * impurity_right)
                 - (self.weighted_n_left / self.weighted_n_node_samples * impurity_left)))

    def children_impurity(self):
        """Evaluate the impurity in children nodes.
        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).
        """

        sample_weight = self.sample_weight
        samples = self.samples
        pos = self.pos
        start = self.start

        sum_left = self.sum_left
        sum_right = self.sum_right

        sq_sum_left = 0.0

        w = 1.0

        for p in range(start, pos):
            i = samples[p]

            if sample_weight is not None:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left = sq_sum_left / self.weighted_n_left
        impurity_right = sq_sum_right / self.weighted_n_right

        for k in range(self.n_outputs):
            impurity_left -= (sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right -= (sum_right[k] / self.weighted_n_right) ** 2.0

        impurity_left /= self.n_outputs
        impurity_right /= self.n_outputs
        return impurity_left, impurity_right


class SplitRecord():
    def __init__(self):
        # Data to track sample split
        self.feature = 0  # Which feature to split on.
        self.pos = 0  # Split samples array at the given position, pos is >= end if the node is a leaf.
        self.threshold = 0  # Threshold to split at.
        self.improvement = 0  # Impurity improvement given parent node.
        self.impurity_left = 0  # Impurity of the left split.
        self.impurity_right = 0  # Impurity of the right split.


# function to find the partition position
def partition(array, samples, low, high):
    # choose the rightmost element as pivot
    pivot = array[high]
    # pointer for greater element
    i = low - 1

    # traverse through all elements
    # compare each element with pivot
    for j in range(low, high):
        if array[j] <= pivot:
            # if element smaller than pivot is found
            # swap it with the greater element pointed by i
            i = i + 1

            # swapping element at i with element at j
            if i != j:
                swap(array, samples, i, j)

    # swap the pivot element with the greater element specified by i
    swap(array, samples, i + 1, high)

    # return the position from where partition is done
    return i + 1


# function to perform quicksort
def quickSort(array, samples, low, high):
    if low < high:
        # find pivot element such that
        # element smaller than pivot are on the left
        # element greater than pivot are on the right
        pi = partition(array, samples, low, high)

        # recursive call on the left of pivot
        quickSort(array, samples, low, pi - 1)

        # recursive call on the right of pivot
        quickSort(array, samples, pi + 1, high)


def swap(Xf, samples, i, j):
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


class BestSplitter():
    """Splitter is called by tree builders to find the best splits, one split at a time.
    """

    def __init__(self, criterion, max_features, min_samples_leaf, min_weight_leaf, random_state,
                 n_splitting_per_node=8):
        """
        Parameters
        ----------
        criterion :
            The criterion to measure the quality of a split.
        max_features :
            The maximal number of randomly selected features which can be
            considered for a split.
        min_samples_leaf :
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.
        min_weight_leaf :
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.
        random_state : object
            The user inputted random state to be used for pseudo-randomness
        """
        self.criterion = criterion

        self.samples = None
        self.n_samples = 0
        self.features = None
        self.n_features = 0
        self.feature_values = None

        self.sample_weight = None

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.n_splitting_per_node = n_splitting_per_node

    def init(self, X, y, sample_weight):
        """Initialize the splitter.
        Take in the input data X, the target Y, and optional sample weights.
        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.
        y : ndarray
            This is the vector of targets, or true labels, for the samples
        sample_weight :
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight.
        """

        self.rand_r_state = self.random_state.randint(0, 100)
        n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        self.samples = np.zeros((n_samples), dtype=np.intp)
        samples = self.samples

        weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight is None or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight is not None:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        n_features = X.shape[1]
        self.features = np.zeros((n_features), dtype=np.intp)
        features = self.features

        for i in range(n_features):
            features[i] = i

        self.n_features = n_features

        self.feature_values = np.zeros((n_samples))

        self.y = y
        self.X = X
        self.sample_weight = sample_weight
        return 0

    def _init_split(self, splitRecord, start_pos):
        inf = 1e5
        splitRecord.impurity_left = inf
        splitRecord.impurity_right = inf
        splitRecord.pos = start_pos
        splitRecord.feature = 0
        splitRecord.threshold = 0.
        splitRecord.improvement = -1 * inf

    def node_reset(self, start, end):
        """Reset splitter on node samples[start:end].
        Parameters
        ----------
        start :
            The index of the first sample to consider
        end :
            The index of the last sample to consider
        return
            The total weight of those samples
        """

        self.start = start
        self.end = end

        self.criterion.init(self.y, self.sample_weight, self.weighted_n_samples, self.samples, start, end)

        return self.criterion.weighted_n_node_samples

    def node_value(self, dest):
        """Copy the value of node samples[start:end] into dest (n_outputs,)."""

        self.criterion.node_value(dest)

    def node_impurity(self):
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()

    def node_split(self, impurity):
        """Find the best split on node samples[start:end].
        The majority of computation will be done here.

        """
        # Find the best split
        samples = self.samples
        start = self.start
        end = self.end

        features = self.features
        n_features = self.n_features

        Xf = self.feature_values
        max_features = self.max_features
        min_samples_leaf = self.min_samples_leaf
        min_weight_leaf = self.min_weight_leaf

        best = SplitRecord()
        current = SplitRecord()
        best_proxy_improvement = -1e5

        f_i = n_features

        n_visited_features = 0

        self._init_split(best, end)

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant nodes.

        while (n_visited_features < max_features):
            n_visited_features += 1

            # Draw a feature at random
            f_j = np.random.randint(0, f_i)

            current.feature = features[f_j]

            # Sort samples along that feature; by copying the values into an array and
            # sorting the array in a manner which utilizes the cache more effectively.
            for i in range(start, end):
                Xf[i] = self.X[samples[i], current.feature]

            quickSort(Xf, samples, start, end - 1)

            f_i -= 1
            features[f_i], features[f_j] = features[f_j], features[f_i]

            # Evaluate all splits
            self.criterion.reset()
            p = start
            if end - start > self.n_splitting_per_node:
                splitingPos = np.sort(
                    np.random.choice(range(start, end), size=self.n_splitting_per_node, replace=False))
            else:
                splitingPos = range(start, end)
            for p in splitingPos:
                p += 1

                if p < end:
                    current.pos = p

                    # Reject if min_samples_leaf is not guaranteed
                    if (((current.pos - start) < min_samples_leaf) or ((end - current.pos) < min_samples_leaf)):
                        continue

                    self.criterion.update(current.pos)

                    # Reject if min_weight_leaf is not satisfied
                    if ((self.criterion.weighted_n_left < min_weight_leaf) or (
                            self.criterion.weighted_n_right < min_weight_leaf)):
                        continue

                    current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        # sum of halves is used to avoid infinite value
                        current.threshold = Xf[p - 1] / 2.0 + Xf[p] / 2.0

                        if ((current.threshold == Xf[p]) or (current.threshold == INFINITY) or (
                                current.threshold == -INFINITY)):
                            current.threshold = Xf[p - 1]

                        best = current  # copy

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            partition_end = end
            p = start

            while p < partition_end:
                if self.X[samples[p], best.feature] <= best.threshold:
                    p += 1

                else:
                    partition_end -= 1
                    samples[p], samples[partition_end] = samples[partition_end], samples[p]

            self.criterion.reset()
            self.criterion.update(best.pos)
            best.impurity_left, best.impurity_right = self.criterion.children_impurity()
            best.improvement = self.criterion.impurity_improvement(impurity, best.impurity_left, best.impurity_right)

        # Return values
        return best


class Node():
    # Base storage structure for the nodes in a Tree object
    def __init__(self):
        self.left_child = None  # id of the left child of the node
        self.right_child = None  # id of the right child of the node
        self.feature = None  # Feature used for splitting the node
        self.threshold = None  # Threshold value at the node
        self.impurity = None  # Impurity of the node (i.e., the value of the criterion)
        self.n_node_samples = None  # Number of samples at the node
        self.weighted_n_node_samples = None  # Weighted number of samples at the node


class Tree():
    """Array-based representation of a binary decision tree.
    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are arbitrary!
    Attributes
    ----------
    node_count : int
        The number of nodes (internal nodes + leaves) in the tree.
    capacity : int
        The current capacity (i.e., size) of the arrays, which is at least as great as `node_count`.
    max_depth : int
        The depth of the tree, i.e. the maximum depth of its leaves.
    """

    def __init__(self, n_features, n_outputs):
        """Constructor."""
        # Input/Output layout
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.value_stride = n_outputs

        # Inner structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = None
        self.nodes = None

    def getNodeValues(self):
        return self.value.reshape(-1, self.n_outputs)

    def getNodeFeatures(self):
        return [node.feature for node in self.nodes]

    def _resize(self, capacity=SIZE_MAX):
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.
        """
        if capacity == self.capacity and self.nodes is not None:
            return 0

        if capacity == SIZE_MAX:
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        if self.nodes is None:
            self.nodes = [Node() for i in range(capacity)]
        else:
            oldNodes = self.nodes
            newNodes = [Node() for i in range(capacity)]
            n_nodes = len(newNodes) if len(oldNodes) > len(newNodes) else len(oldNodes)
            for i in range(n_nodes):
                newNodes[i] = oldNodes[i]
            self.nodes = newNodes

        if self.value is None:
            self.value = np.zeros((capacity * self.value_stride))
        else:
            oldValues = self.value
            newValues = np.zeros((capacity * self.value_stride))
            n_values = len(newValues) if len(newValues) < len(oldValues) else len(oldValues)
            for i in range(n_values):
                newValues[i] = oldValues[i]
            self.value = newValues

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    def _add_node(self, parent, is_left, is_leaf, feature, threshold, impurity, n_node_samples,
                  weighted_n_node_samples):
        """Add a node to the tree.
        The new node registers itself as the child of its parent.
        Returns (size_t)(-1) on error.
        """
        node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize() != 0:
                return SIZE_MAX

        node = self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED

        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold

        self.node_count += 1

        return node_id

    def predict(self, X):
        """Predict target for X."""
        destList = self.apply(X)
        values = self._get_value_ndarray()
        out = np.zeros((X.shape[0], self.n_outputs))
        for i in range(len(destList)):
            out[i] = values[destList[i]]

        return out

    def apply(self, X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        # Check input
        assert isinstance(X, np.ndarray), "X should be in np.ndarray format, got %s" % type(X)

        # Extract input
        X_ndarray = X
        n_samples = X.shape[0]

        # Initialize output
        out = np.zeros((n_samples,), dtype=np.intp)

        for i in range(n_samples):
            node = self.nodes[0]
            leafDest = 0
            # While node not a leaf
            while node.left_child != _TREE_LEAF:
                if X_ndarray[i, node.feature] <= node.threshold:
                    leafDest = node.left_child
                    node = self.nodes[node.left_child]
                else:
                    leafDest = node.right_child
                    node = self.nodes[node.right_child]

            out[i] = leafDest  # node offset

        return out

    def _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array.
        The array keeps a reference to this Tree, which manages the underlying memory.
        """
        return self.value.reshape(self.node_count, self.n_outputs)

    # =============================================================================


# PriorityHeap data structure
# =============================================================================
class PriorityHeapRecord():
    def __init__(self):
        self.node_id = -1
        self.start = -1
        self.end = -1
        self.pos = -1
        self.depth = -1
        self.is_leaf = -1
        self.impurity = -1.0
        self.impurity_left = -1.0
        self.impurity_right = -1.0
        self.improvement = -1.0


class PriorityHeap():
    """A priority queue implemented as a binary heap.
    The heap invariant is that the impurity improvement of the parent record
    is larger than the impurity improvement of the children.
    Attributes
    ----------
    capacity : int
        The capacity of the heap
    heap_ptr : int
        The water mark of the heap; the heap grows from left to right in the
        array ``heap_``.
    heap_ : PriorityHeapRecord
        The array of heap records. The maximum element is on the left;
        the heap grows from left to right
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.heap_ptr = 0
        self.heap_ = [PriorityHeapRecord() for i in range(capacity)]

    def is_empty(self):
        return self.heap_ptr <= 0

    def heapify_up(self, heap, pos):
        """Restore heap invariant parent.improvement > child.improvement from
           ``pos`` upwards.
        heap : array of PriorityHeapRecord
        """
        if pos == 0:
            return

        parent_pos = int((pos - 1) / 2)

        if heap[parent_pos].improvement < heap[pos].improvement:
            heap[parent_pos], heap[pos] = heap[pos], heap[parent_pos]
            self.heapify_up(heap, parent_pos)

    def heapify_down(self, heap, pos, heap_length):
        """Restore heap invariant parent.improvement > children.improvement from
           ``pos`` downwards.
        heap : array of PriorityHeapRecord
        """
        left_pos = 2 * (pos + 1) - 1
        right_pos = 2 * (pos + 1)
        largest = pos

        if (left_pos < heap_length and heap[left_pos].improvement > heap[largest].improvement):
            largest = left_pos

        if (right_pos < heap_length and heap[right_pos].improvement > heap[largest].improvement):
            largest = right_pos

        if largest != pos:
            heap[pos], heap[largest] = heap[largest], heap[pos]
            self.heapify_down(heap, largest, heap_length)

    def push(self, node_id, start, end, pos, depth, is_leaf, improvement, impurity, impurity_left, impurity_right):
        """Push record on the priority heap.
        """
        heap_ptr = self.heap_ptr

        # Resize if capacity not sufficient
        if heap_ptr >= self.capacity:
            self.capacity *= 2
            oldHeap = self.heap_
            newHeap = [PriorityHeapRecord() for i in range(self.capacity)]
            for i in range(len(oldHeap)):
                newHeap[i] = oldHeap[i]
            self.heap_ = newHeap

        # Put element as last element of heap
        heap = self.heap_  # array of PriorityHeapRecord
        heap[heap_ptr].node_id = node_id
        heap[heap_ptr].start = start
        heap[heap_ptr].end = end
        heap[heap_ptr].pos = pos
        heap[heap_ptr].depth = depth
        heap[heap_ptr].is_leaf = is_leaf
        heap[heap_ptr].impurity = impurity
        heap[heap_ptr].impurity_left = impurity_left
        heap[heap_ptr].impurity_right = impurity_right
        heap[heap_ptr].improvement = improvement

        # Heapify up
        self.heapify_up(heap, heap_ptr)

        # Increase element count
        self.heap_ptr = heap_ptr + 1
        return 0

    def pop(self):
        """Remove max element from the heap.
        """
        heap_ptr = self.heap_ptr
        heap = self.heap_

        if heap_ptr <= 0:
            return -1

        # Take first element
        res = heap[0]

        # Put last element to the front
        heap[0], heap[heap_ptr - 1] = heap[heap_ptr - 1], heap[0]

        # Restore heap invariant
        if heap_ptr > 1:
            self.heapify_down(heap, 0, heap_ptr - 1)

        self.heap_ptr = heap_ptr - 1

        return res


class BestFirstTreeBuilder():
    """Build a decision tree in best-first fashion.
    The best node to expand is given by the node at the frontier that has the highest impurity improvement.
    """

    def __init__(self, splitter, min_samples_split, min_samples_leaf, min_weight_leaf, max_depth, max_leaf_nodes,
                 min_impurity_decrease):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

    def _add_to_frontier(self, rec, frontier):
        """Adds record ``rec`` to the priority queue ``frontier``
        rec : PriorityHeapRecord
        frontier: PriorityHeap
        """
        return frontier.push(rec.node_id, rec.start, rec.end, rec.pos, rec.depth, rec.is_leaf, rec.improvement,
                             rec.impurity, rec.impurity_left, rec.impurity_right)

    def build(self, tree, X, y, sample_weight=None):
        """Build a decision tree from the training set (X, y)."""

        # Parameters
        splitter = self.splitter
        max_leaf_nodes = self.max_leaf_nodes

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight)

        INITIAL_STACK_SIZE = 10
        frontier = PriorityHeap(INITIAL_STACK_SIZE)
        split_node_left = PriorityHeapRecord()
        split_node_right = PriorityHeapRecord()

        n_node_samples = splitter.n_samples
        max_split_nodes = max_leaf_nodes - 1

        max_depth_seen = -1

        # Initial capacity
        init_capacity = max_split_nodes + max_leaf_nodes
        print("init_capacity:", init_capacity)
        tree._resize(init_capacity)

        # add root to frontier
        rc = self._add_split_node(splitter, tree, 0, n_node_samples, INFINITY, IS_FIRST, IS_LEFT, None, 0,
                                  split_node_left)
        if rc >= 0:
            rc = self._add_to_frontier(split_node_left, frontier)

        assert rc != -1, "MemoryError in BestFirstTreeBuilder -> build -> _add_split_node"

        while not frontier.is_empty():
            record = frontier.pop()

            node = tree.nodes[record.node_id]
            is_leaf = (record.is_leaf or max_split_nodes <= 0)

            if is_leaf:
                # Node is not expandable; set node as leaf
                node.left_child = _TREE_LEAF
                node.right_child = _TREE_LEAF
                node.feature = _TREE_UNDEFINED
                node.threshold = _TREE_UNDEFINED

            else:
                # Node is expandable

                # Decrement number of split nodes available
                max_split_nodes -= 1

                # Compute left split node
                rc = self._add_split_node(splitter, tree, record.start, record.pos, record.impurity_left,
                                          IS_NOT_FIRST, IS_LEFT, record.node_id, record.depth + 1, split_node_left)
                if rc == -1:
                    break

                # tree.nodes may have changed
                node = tree.nodes[record.node_id]

                # Compute right split node
                rc = self._add_split_node(splitter, tree, record.pos, record.end, record.impurity_right, IS_NOT_FIRST,
                                          IS_NOT_LEFT, record.node_id, record.depth + 1, split_node_right)
                if rc == -1:
                    break

                # Add nodes to queue
                rc = self._add_to_frontier(split_node_left, frontier)
                if rc == -1:
                    break

                rc = self._add_to_frontier(split_node_right, frontier)
                if rc == -1:
                    break

            if record.depth > max_depth_seen:
                max_depth_seen = record.depth

        if rc >= 0:
            rc = tree._resize(tree.node_count)

        if rc >= 0:
            tree.max_depth = max_depth_seen

        assert rc != -1, "MemoryError in BestFirstTreeBuilder -> build"

    def _add_split_node(self, splitter, tree, start, end, impurity, is_first, is_left, parent, depth, res):
        """Adds node w/ partition ``[start, end)`` to the frontier.
        parent : node id
        res : PriorityHeapRecord
        """
        split = SplitRecord()

        min_impurity_decrease = self.min_impurity_decrease

        weighted_n_node_samples = splitter.node_reset(start, end)

        if is_first:
            impurity = splitter.node_impurity()

        n_node_samples = end - start
        is_leaf = (depth >= self.max_depth or n_node_samples < self.min_samples_split or
                   n_node_samples < 2 * self.min_samples_leaf or
                   weighted_n_node_samples < 2 * self.min_weight_leaf or
                   impurity <= EPSILON  # impurity == 0 with tolerance
                   )

        if not is_leaf:
            split = splitter.node_split(impurity)
            # If EPSILON=0 in the below comparison, float precision issues stop
            # splitting early, producing trees that are dissimilar to v0.18
            is_leaf = (is_leaf or split.pos >= end or split.improvement + EPSILON < min_impurity_decrease)

        node_id = tree._add_node(parent if parent is not None else _TREE_UNDEFINED, is_left, is_leaf,
                                 split.feature, split.threshold, impurity, n_node_samples, weighted_n_node_samples)
        if node_id == SIZE_MAX:
            return -1

        # compute values also for split nodes (might become leafs later).
        tempValue = np.zeros((tree.value_stride))
        splitter.node_value(tempValue)
        tree.value[node_id * tree.value_stride:node_id * tree.value_stride + tree.value_stride] = tempValue

        res.node_id = node_id
        res.start = start
        res.end = end
        res.depth = depth
        res.impurity = impurity

        if not is_leaf:
            # is split node
            res.pos = split.pos
            res.is_leaf = 0
            res.improvement = split.improvement
            res.impurity_left = split.impurity_left
            res.impurity_right = split.impurity_right

        else:
            # is leaf => 0 improvement
            res.pos = end
            res.is_leaf = 1
            res.improvement = 0.0
            res.impurity_left = impurity
            res.impurity_right = impurity

        return 0


class BaseDecisionTree():

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features=None, max_leaf_nodes=None, random_state=47, min_impurity_decrease=0.0):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = None
        self._is_fitted__ = False

    def check_random_state(self, seed):
        assert isinstance(seed, numbers.Integral), "The random seed shoulb be an integer!"
        return np.random.RandomState(seed)

    def check_sample_weight(self, sample_weight, n_samples):
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float64)
        else:
            assert sample_weight.ndim == 1, "sample_weight must be 1D array!"
            assert sample_weight.shape == (n_samples,), "sample_weight.shape == {}, expected {}!".format(
                sample_weight.shape, (n_samples,))
        return sample_weight

    def check_is_fitted(self):
        if self._is_fitted__ == False:
            raise AttributeError('Error! This decision tree is not fitted!')

    def get_depth(self):
        """Return the depth of the decision tree: the maximum distance between the root
        and any leaf.
        self.tree_.max_depth : int
        """
        self.check_is_fitted()
        return self.tree_.max_depth

    def fit(self, X, y, sample_weight=None):
        random_state = self.check_random_state(self.random_state)

        # Determine output settings
        n_samples, self.n_features_in_ = X.shape

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]  # n_classes; or 1 if only 1 target number is given for each sample

        # Check parameters
        max_depth = 1e5 if self.max_depth is None else self.max_depth
        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

        assert isinstance(self.min_samples_leaf, numbers.Integral), "min_samples_leaf must be an Integer!"
        assert self.min_samples_leaf >= 1, "min_samples_leaf must be at least 1, got %s" % self.min_samples_leaf
        min_samples_leaf = self.min_samples_leaf

        assert isinstance(self.min_samples_split, numbers.Integral), "min_samples_split must be an Integer!"
        assert self.min_samples_split >= 2, "min_samples_split must be at least 2, got %s" % self.min_samples_split
        min_samples_split = self.min_samples_split
        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        max_features = self.n_features_in_
        self.max_features_ = max_features

        assert len(y) == n_samples, "Number of labels=%d does not match number of samples=%d" % (len(y), n_samples)

        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        assert max_depth > 0, "max_depth must be greater than zero."

        assert isinstance(max_leaf_nodes,
                          numbers.Integral), "max_leaf_nodes must be integral number but was %r" % max_leaf_nodes
        if max_leaf_nodes != -1:
            assert max_leaf_nodes >= 2, "max_leaf_nodes {} must be larger than 1".format(max_leaf_nodes)

        sample_weight = self.check_sample_weight(sample_weight, n_samples)

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * np.sum(sample_weight)

        assert self.min_impurity_decrease >= 0.0, "min_impurity_decrease must be greater than or equal to 0!"

        # Build tree
        criterion = MSERegressionCriterion(self.n_outputs_, n_samples)

        splitter = BestSplitter(criterion, self.max_features_, min_samples_leaf, min_weight_leaf, random_state)

        print("self.n_outputs_, n_samples, self.n_features_in_, max_leaf_nodes", self.n_outputs_, n_samples,
              self.n_features_in_, max_leaf_nodes)
        self.tree_ = Tree(self.n_features_in_, self.n_outputs_)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        assert max_leaf_nodes > 0, "You must provide the max_leaf_nodes parameter (>0)"
        builder = BestFirstTreeBuilder(splitter, min_samples_split, min_samples_leaf,
                                       min_weight_leaf, max_depth, max_leaf_nodes, self.min_impurity_decrease)

        builder.build(self.tree_, X, y, sample_weight)

        self._is_fitted__ = True

        return self

    def _validate_X_predict(self, X):
        """Validate the training data on predict (probabilities)."""
        self.n_features_in_ = X.shape[1]
        return X

    def predict(self, X):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.
        Parameters
        ----------
        X : The input samples of shape (n_samples, n_features).
        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        self.check_is_fitted()
        X = self._validate_X_predict(X)
        proba = self.tree_.predict(X)
        return proba

    def apply(self, X):
        """Return the index of the leaf that each sample is predicted as.
        Parameters
        ----------
        X : The input samples of shape (n_samples, n_features).
        Returns
        -------
        X_leaves : array-like of shape (n_samples,)
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
        """
        self.check_is_fitted()
        X = self._validate_X_predict(X)
        return self.tree_.apply(X)


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

        self.predict_proba = None  # the prediction for the instance being explained

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

        ###############################################################
        # ridge = RidgeSGD(alpha=1, fit_intercept=True)
        # ridge.ridgeSGDFit(neighborhood_data[:, used_features], labels_column, sample_weight=weights)

        # if self.verbose:
        #     ridge.printFittedModel(len(used_features))
        # return ridge.coef_, ridge.intercept_, ridge.n_iter_

        myTree = BaseDecisionTree(max_leaf_nodes=60)
        myTree.fit(neighborhood_data[:, used_features], neighborhood_labels, sample_weight=weights)

        print("myTree.NodeValues:", myTree.tree_.getNodeValues())
        print("myTree.NodeFeatures:", myTree.tree_.getNodeFeatures())
        print("myTree.NodeCount:", myTree.tree_.node_count)

        ###############################################################


def resetRandomStates(manualseed=47):
    random.seed(manualseed)
    torch.manual_seed(manualseed)
    np.random.seed(manualseed)


def readOriginalTrainData(dataPath):
    _, _, _, _, entireDataset = readPrivateTrainingData(dataPath, binaryClassification=True)
    training_data = entireDataset.getNumpyData()
    training_labels = entireDataset.getNumpyLabels()
    print("training_data.shape: {}; training_data.shape: {}".format(training_data.shape, training_labels.shape))
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


if __name__ == "__main__":
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

    data, inverse = loadRegressiveData(dataPath=parameters['regressiveDataPath'],
                                       inversePath=parameters['regressiveInversePath'])

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

