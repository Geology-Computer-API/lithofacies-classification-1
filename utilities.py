import numpy as np
from sklearn.model_selection import cross_val_score
from scipy.stats.stats import pearsonr
from sklearn.metrics import mutual_info_score
import pandas as pd
from scipy.sparse import issparse
import matplotlib.pyplot as plt

def read_dataset():
    dFrame = pd.read_csv('../datasets/facies_vectors.csv',
                         dtype={'Facies': int, 'Formation': str, 'Well Name': str, 'Depth': float, 'GR': float,
                                'ILD_log10': float, 'DeltaPHI': float, 'PHIND': float, 'PE': float, 'NM_M': int,
                                'RELPOS': float})
    features = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
    feature_matrix = dFrame[features].as_matrix()
    facies = ['Facies']
    facies_vector = dFrame[facies].as_matrix()
    facies_vector = facies_vector.reshape((4149,))
    return feature_matrix, facies_vector


def convert_to_binary_classification(x, facies_vector):
    # class 1 means it is class x in the original problem
    facies_vector_2_over_rest = [1 if i == x else 0 for i in facies_vector]
    return np.array(facies_vector_2_over_rest)


def draw_data_histogram(facies_vector):
    # distribution = np.bincount(facies_vector)
    bins = np.linspace(1, 9, 20)
    plt.hist(facies_vector, bins=bins, align='mid')
    plt.title("Class Distribution Histogram")
    plt.xlabel("Class")
    plt.ylabel("Frequency")


def get_feature_statistics(feature_matrix):
    pass





# ENSEMBLE CLASS
class ensemble_clfs:
    def __init__(self, clf_list):
        self.clf_list = clf_list
        self.n_clfs = len(clf_list)
        self.trained_clfs = [None] * self.n_clfs
        self.trained_ids = []

    def fit(self, X, y, clf_id):
        clf = self.clf_list[clf_id]
        clf.fit(X, y)
        self.trained_clfs[clf_id] = clf
        self.trained_ids += [clf_id]

    def predict(self, X):
        n_trained = len(self.trained_clfs)
        pred_list = np.zeros((X.shape[0], n_trained, 9))#np.max(self.trained_ids)+1
        for i in self.trained_ids:
            clf = self.trained_clfs[i]
            y_pred = clf.predict_proba(X)
            pred_list[:, i, :] = y_pred
        return np.mean(pred_list, axis=1)


#### FEATURE SELECTION
def compute_feature_curve(clf, X, y, ft_ranks, step_size=1, score_name="f1_micro"):
    """plots learning curve """
    selected_features = []
    scores = []

    n_features = X.shape[1]

    if score_name == "f1_micro":
        score_function = "f1_micro"

    elif score_name == "f1_macro":
        score_function = "f1_macro"

    for ft_list in range(step_size, n_features + 1, step_size):
        score = np.mean(cross_val_score(clf, X[:, ft_ranks[:ft_list]], y, 
                                        cv=10, scoring=score_function))
        
        selected_features += [ft_list]
        scores += [score]

        print('%s score: %.3f with %s features...' % (score_name, score, ft_list))

    print('Best score achieved : %.3f \n' % np.amax(scores))
    return (scores, selected_features)


def greedy_selection(clf, X, y, score_name="f1_micro"):
    """Applies greedy forward selection"""
    n_features = X.shape[1]

    global_max = 0.0
    selected_features = []

    if score_name == "f1_micro":
        score_function = "f1_micro"

    elif score_name == "f1_macro":
        score_function = "f1_macro"

    scores = []

    for i in range(n_features):
        maximum = 0.0
        for j in range(n_features):
            if j in selected_features:
                continue

            score = np.mean(cross_val_score(
                            clf, X[:, selected_features + [j]], y, cv=4,
                            scoring=score_function))
            
            if score > maximum:
                maximum = score
                best_feature = j

        scores += [score]
        selected_features += [best_feature]

        print('%s score: %.3f with features: %s ...' % (score_name,
                                                        score,
                                                        selected_features))

        if maximum > global_max:
            global_max = maximum

    return scores, np.arange(len(selected_features)) + 1


def rank_features(X, y, corr='fisher'):
    """returns ranked indices using a correlation function
    """
    correlation_functions = {
        'fisher': fisher_crit,
        'mutual_info': mutual_info_score,
        'info_gain': information_gain
    }

    results = []

    n_features = X.shape[1]

    if corr in ['pearson']:
        for feature in range(n_features):
            results.append((feature, abs(pearsonr(X[:, feature], y)[0])))

    elif corr in ["fisher"]:
        for feature in range(n_features):
            results.append(
                (feature, correlation_functions[corr](X[:, feature], y)))

    results = sorted(results, key=lambda a: -a[1])

    rank_list = [f[0] for f in results]
    scores = [1 if np.isnan(f[1]) else f[1] for f in results]

    return rank_list, scores

#### MISC
def mapit(vector):

    s = np.unique(vector)

    mapping = pd.Series([x[0] for x in enumerate(s)], index = s)
    vector=vector.map(mapping)
    return vector

def fisher_crit(v1, v2):
    """computes the fisher's criterion"""
    if issparse(v1):
        v1 = v1.todense()
    return abs(np.mean(v1) - np.mean(v2)) / (np.var(v1) + np.var(v2))


def information_gain(v1, v2):
    """computes the information gain"""
    if issparse(v1):
        v1 = v1.todense()
    return abs(np.mean(v1) - np.mean(v2)) / (np.var(v1) + np.var(v2))