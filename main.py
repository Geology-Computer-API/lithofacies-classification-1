import sys
import argparse
import methods
import numpy as np
import pylab as pl
import preprocessing
import utilities as ut


if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()

    parser.add_argument('-fs', '--fs_functions', nargs="+", default="pearson",
                        choices=["pearson", "fisher", "greedy"])

    parser.add_argument('-m', '--method', default="forward_selection",
                        choices=["forward_selection",
                                 "ensemble_svm",
                                 "ensemble_heter"])

    parser.add_argument('-n', '--n_clfs', default=5, type=int)

    parser.add_argument('-p', '--problem', default="multiclassification",
                        choices=["multiclassification", "ovr"])
    parser.add_argument('-u', '--unbalanced', default=False)

    args = parser.parse_args()
    method = args.method
    fs_functions = args.fs_functions
    n_clfs = args.n_clfs
    problem = args.problem
    use_unbalanced_data = args.unbalanced
    np.random.seed(1)

    # GET DATASET
    feature_matrix, facies_vector = ut.read_dataset()
    # draw_data_histogram(facies_vector)
    ut.get_feature_statistics(feature_matrix)

    # Preprocessing
    feature_matrix = preprocessing.normalize(feature_matrix, facies_vector)

    # convert the dataset to be a binary classification problem for class x over rest
    if problem == "ovr":
        facies_vector = ut.convert_to_binary_classification(2, facies_vector)

    if not use_unbalanced_data:
        feature_matrix, facies_vector = preprocessing.balance(feature_matrix, facies_vector)

    # RUN TRANING METHOD
    print("Method: ", method)

    methods.run_method(method, feature_matrix, facies_vector, n_clfs=n_clfs,
                       fs_functions=fs_functions)

    pl.legend(loc="best")
    pl.show()


