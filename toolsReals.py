from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from dask.distributed import Client
import joblib


class BestClModel:

    def __init__(self, model='svm'):
        """
        :param model:
        STR dytpe

        (FUTURE PROJECT: ADD MODELS):

        possibilities:
        svm:  Support Vector Machine (default)
        logreg: Logistic Regression
        dt: Decision Tree
        rf: Random Forest
        knn: KNearestNeighbors
        """
        self.model = model

    def metrics_res(self, y_true, y_pred):
        print('Metrics results:')
        print('Accuracy: ' + str(accuracy_score(y_true, y_pred)))
        print('Recall: ' + str(recall_score(y_true, y_pred)))
        print('F1 Score: ' + str(f1_score(y_true, y_pred)))

    def conf_matrix(self, y_true, y_pred, n_classes):
        conf = confusion_matrix(y_true, y_pred).ravel().reshape(n_classes, n_classes)
        _ = sns.heatmap(conf, annot=True, fmt='g')

    def best_svm(self, X_train, y_train, X_test):
        client = Client(processes=False)
        params = {'C': [1, 2, 4, 5, 7],
                  'kernel': ['rbf', 'sigmoid'],
                  'gamma': ['scale', 'auto'],
                  'class_weight': [None, 'balanced'],
                  'decision_function_shape': ['ovr', 'ovo']
                  }

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        svc = SVC()
        search = GridSearchCV(svc, params, verbose=10, n_jobs=-1)

        with joblib.parallel_backend('dask'):
            search.fit(X_train, y_train)
        print('\nBest parameters: ', search.best_params_)

        return search.predict(X_test)
