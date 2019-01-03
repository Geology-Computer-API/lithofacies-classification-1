import utilities as ut
import performance_summary
import numpy as np
from sklearn.svm import SVC
import pylab as plt
from sklearn import model_selection
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from performance_summary import get_accuracy_within_one_facies

def run_method(method, X, y, n_clfs=5, fs_functions="pearson", score_name="f1_micro"):

    if method == "forward_selection":
        """
        Forward selection using weighted svm w.r.t greedy, pearson and fisher
        """
        w_svm = SVC(class_weight='balanced', probability=True)

        for fs in fs_functions:
            print("FEATURE SELECTION: %s\n" % fs)

            # GET FEATURES RANK
            if fs in ["pearson", "fisher"]:
                print("Ranking features using %s ..." % fs)
                ft_ranks, scores = ut.rank_features(np.array(X), y, corr=fs)

                scores, selected_features = ut.compute_feature_curve(w_svm, X, y,
                                                                     ft_ranks=ft_ranks,
                                                                     step_size=1,
                                                                     score_name=score_name)
                print (selected_features)

            elif fs == "greedy":
                scores, selected_features = ut.greedy_selection(w_svm, X, y, score_name=score_name)

            plt.plot(selected_features, scores, label=fs)

        plt.xlabel("Number of retained features")

    elif method == "ensemble_svm":
        clfs = []
        for c in [1, 10, 100, 500, 1000]:
            #for g in [1.00000000e-02, 1.00000000e-01, 1.00000000e+00, 1.00000000e+01, 1.00000000e+02]:
            clfs += [SVC(probability=True, C=c, gamma=10.0, class_weight='balanced')]

        (scores_dict, x_values) = ensemble_forward_pass(clfs, X, y, n_clfs=len(clfs))
        plt.plot(x_values, scores_dict["accuracy"], label="SVM ensemble, accuracy")
        plt.plot(x_values, scores_dict["accuracy_within_one_facies"], label="SVM ensemble, accuracy_within_one_facies")
        plt.plot(x_values, scores_dict["f1_micro"], label="SVM ensemble, f1_micro")
        plt.plot(x_values, scores_dict["f1_macro"], label="SVM ensemble, f1_macro")
        plt.plot(x_values, scores_dict["f1_weighted"], label="SVM ensemble, f1_weighted")
        plt.plot(x_values, scores_dict["precision_micro"], label="SVM ensemble, precision_micro")
        plt.plot(x_values, scores_dict["precision_macro"], label="SVM ensemble, precision_macro")
        plt.plot(x_values, scores_dict["precision_weighted"], label="SVM ensemble, precision_weighted")
        plt.plot(x_values, scores_dict["recall_micro"], label="SVM ensemble, recall_micro")
        plt.plot(x_values, scores_dict["recall_macro"], label="SVM ensemble, recall_macro")
        plt.plot(x_values, scores_dict["recall_weighted"], label="SVM ensemble, recall_weighted")

    elif method == "ensemble_heter":
        """
        Description in section 5.4 - Results in Fig. 11
        """
        clfs = [SVC(probability=True, gamma=1.0, C=10, class_weight='balanced', kernel='rbf'), GaussianNB(),
                RandomForestClassifier(n_estimators=20),
                GradientBoostingClassifier(n_estimators=17, learning_rate=0.3,max_depth=13),
                SGDClassifier(alpha=.0001, loss='log', max_iter=50,penalty="elasticnet"),
                LogisticRegression(penalty='l2', C=10, class_weight='balanced',
                solver='newton-cg', multi_class='ovr', max_iter=600)]

        # clfs = [SVC(probability=True, gamma=10.0, C=10, class_weight='balanced', kernel='rbf'), GaussianNB(),
        #         RandomForestClassifier(n_estimators=20),
        #         GradientBoostingClassifier(n_estimators=17, learning_rate=0.3,max_depth=13),
        #         SGDClassifier(alpha=.0001, loss='log', max_iter=50,penalty="elasticnet"),
        #         LogisticRegression(penalty='l2', C=10, class_weight='balanced',
        #         solver='newton-cg', multi_class='ovr', max_iter=400)]

        (scores_dict, x_values) = ensemble_forward_pass(clfs, X, y, n_clfs=n_clfs)
        plt.plot(x_values, scores_dict["accuracy"], label="Ensemble heterogeneous, accuracy")
        plt.plot(x_values, scores_dict["accuracy_within_one_facies"], label="Ensemble heterogeneous, accuracy_within_one_facies")
        plt.plot(x_values, scores_dict["f1_micro"], label="Ensemble heterogeneous, f1_micro")
        plt.plot(x_values, scores_dict["f1_macro"], label="Ensemble heterogeneous, f1_macro")
        plt.plot(x_values, scores_dict["f1_weighted"], label="Ensemble heterogeneous, f1_weighted")
        plt.plot(x_values, scores_dict["precision_micro"], label="Ensemble heterogeneous, precision_micro")
        plt.plot(x_values, scores_dict["precision_macro"], label="Ensemble heterogeneous, precision_macro")
        plt.plot(x_values, scores_dict["precision_weighted"], label="Ensemble heterogeneous, precision_weighted")
        plt.plot(x_values, scores_dict["recall_micro"], label="Ensemble heterogeneous, recall_micro")
        plt.plot(x_values, scores_dict["recall_macro"], label="Ensemble heterogeneous, recall_macro")
        plt.plot(x_values, scores_dict["recall_weighted"], label="Ensemble heterogeneous, recall_weighted")

    else:
        print("%s does not exist..." % method)


#### ENSEMBLE FORWARD PASS
def ensemble_forward_pass(clfs, X, y, n_clfs=None):
    if n_clfs is None:
        n_clfs = len(clfs)
    else:
        n_clfs = n_clfs + 1

    clf_list = ut.ensemble_clfs(clfs)
    scores_dict = {
        "accuracy": np.zeros(n_clfs),
        "accuracy_within_one_facies": np.zeros(n_clfs),
        "f1_micro": np.zeros(n_clfs),
        "f1_macro": np.zeros(n_clfs),
        "f1_weighted": np.zeros(n_clfs),
        "precision_micro": np.zeros(n_clfs),
        "precision_macro": np.zeros(n_clfs),
        "precision_weighted": np.zeros(n_clfs),
        "recall_micro": np.zeros(n_clfs),
        "recall_macro": np.zeros(n_clfs),
        "recall_weighted": np.zeros(n_clfs)
    }

    for i in range(n_clfs):
        skf = model_selection.StratifiedKFold(n_splits=10)

        # CROSS VALIDATE
        accuracy_scores = []
        accuracy_within_one_facies_scores = []
        f1_micro_scores = []
        f1_macro_scores = []
        f1_weighted_scores = []
        precision_micro_scores = []
        precision_macro_scores = []
        precision_weighted_scores = []
        recall_micro_scores = []
        recall_macro_scores = []
        recall_weighted_scores = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf_list.fit(X_train, y_train, i)
            y_pred = clf_list.predict(X_test)

            accuracy_scores += [metrics.accuracy_score(y_test, np.argmax(y_pred, axis=1) + 1, normalize=True)]
            accuracy_within_one_facies_scores += [get_accuracy_within_one_facies(y_test, np.argmax(y_pred, axis=1) + 1)]
            f1_micro_scores += [metrics.f1_score(y_test, np.argmax(y_pred, axis=1) + 1, average='micro')]
            f1_macro_scores += [metrics.f1_score(y_test, np.argmax(y_pred, axis=1) + 1, average='macro')]
            f1_weighted_scores += [metrics.f1_score(y_test, np.argmax(y_pred, axis=1) + 1, average='weighted')]
            precision_micro_scores += [metrics.precision_score(y_test, np.argmax(y_pred, axis=1) + 1, average='micro')]
            precision_macro_scores += [metrics.precision_score(y_test, np.argmax(y_pred, axis=1) + 1, average='macro')]
            precision_weighted_scores += [metrics.precision_score(y_test, np.argmax(y_pred, axis=1) + 1, average='weighted')]
            recall_micro_scores += [metrics.recall_score(y_test, np.argmax(y_pred, axis=1) + 1, average='micro')]
            recall_macro_scores += [metrics.recall_score(y_test, np.argmax(y_pred, axis=1) + 1, average='macro')]
            recall_weighted_scores += [metrics.recall_score(y_test, np.argmax(y_pred, axis=1) + 1, average='weighted')]

        scores_dict["accuracy"][i] = np.mean(accuracy_scores)
        scores_dict["accuracy_within_one_facies"][i] = np.mean(accuracy_within_one_facies_scores)
        scores_dict["f1_micro"][i] = np.mean(f1_micro_scores)
        scores_dict["f1_macro"][i] = np.mean(f1_macro_scores)
        scores_dict["f1_weighted"][i] = np.mean(f1_weighted_scores)
        scores_dict["precision_micro"][i] = np.mean(precision_micro_scores)
        scores_dict["precision_macro"][i] = np.mean(precision_macro_scores)
        scores_dict["precision_weighted"][i] = np.mean(precision_weighted_scores)
        scores_dict["recall_micro"][i] = np.mean(recall_micro_scores)
        scores_dict["recall_macro"][i] = np.mean(recall_macro_scores)
        scores_dict["recall_weighted"][i] = np.mean(recall_weighted_scores)

        print("accuracy_scores: %.3f, n_clfs: %d" % (scores_dict["accuracy"][i], i + 1))
        print("accuracy_within_one_facies_scores: %.3f, n_clfs: %d" % (scores_dict["accuracy_within_one_facies"][i], i + 1))
        print("f1_micro_scores: %.3f, n_clfs: %d" % (scores_dict["f1_micro"][i], i + 1))
        print("f1_macro_scores: %.3f, n_clfs: %d" % (scores_dict["f1_macro"][i], i + 1))
        print("f1_weighted_scores: %.3f, n_clfs: %d" % (scores_dict["f1_weighted"][i], i + 1))
        print("precision_micro_scores: %.3f, n_clfs: %d" % (scores_dict["precision_micro"][i], i + 1))
        print("precision_macro_scores: %.3f, n_clfs: %d" % (scores_dict["precision_macro"][i], i + 1))
        print("precision_weighted_scores: %.3f, n_clfs: %d" % (scores_dict["precision_weighted"][i], i + 1))
        print("recall_micro_scores: %.3f, n_clfs: %d" % (scores_dict["recall_micro"][i], i + 1))
        print("recall_macro_scores: %.3f, n_clfs: %d" % (scores_dict["recall_macro"][i], i + 1))
        print("recall_weighted_scores: %.3f, n_clfs: %d" % (scores_dict["recall_weighted"][i], i + 1))
        print("*******************************************************************************************************")

    y_pred = clf_list.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1) + 1
    print("Done prediction")
    performance_summary.plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix")
    return scores_dict, np.arange(n_clfs) + 1


