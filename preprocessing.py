from sklearn import preprocessing
from imblearn.over_sampling import SMOTE


def normalize(x, y):
    imputer = preprocessing.Imputer(strategy = "median")
    x = imputer.fit_transform(x,y)
    normalized_x = preprocessing.scale(x, axis=0)
    return normalized_x


def balance(x,y):
    return SMOTE().fit_sample(x, y)

