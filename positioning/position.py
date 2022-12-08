import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


class Position_KNN():
    def __init__(self, k=1, metric='euclidean', weight='distance'):
        self.classifier_building = None
        self.classifier_floor = None
        self.regressor = None
        self.y_train = None
        self.k = k
        self.metric = metric
        self.weight = weight

    def fit(self, X_train=None, y_train=None):
        self.y_train = y_train
        self.regressor = KNeighborsRegressor(n_neighbors=self.k, metric=self.metric, weights=self.weight)
        self.classifier_building = KNeighborsClassifier(n_neighbors=self.k, metric=self.metric, weights=self.weight)
        self.classifier_floor = KNeighborsClassifier(n_neighbors=self.k, metric=self.metric, weights=self.weight)
        self.regressor.fit(X_train, y_train[:, 0:3])
        self.classifier_floor .fit(X_train, y_train[:, 3])
        self.classifier_building.fit(X_train, y_train[:, 4])

    def predict_position_2D(self, X_test=None, y_test=None):
        accuracy, true_false_values = self.floor_hit_rate(X_test=X_test, y_test=y_test)
        mask = (true_false_values[:, 0] == 0)
        prediction_2D = self.regressor.predict(X_test)
        euclidean_distance = np.linalg.norm(prediction_2D[:, 0:2] - y_test[:, 0:2], axis=1)
        selected = euclidean_distance[mask]
        mean_error = np.mean(selected)
        return mean_error

    def predict_position_3D(self, X_test=None, y_test=None):
        prediction_3D = self.regressor.predict(X_test)
        mean_error = np.mean(np.linalg.norm(prediction_3D[:, 0:3] - y_test[:, 0:3], axis=1))
        return mean_error

    def floor_hit_rate(self, X_test=None, y_test=None):
        prediction_floor = self.classifier_floor.predict(X_test)
        cm_floor = confusion_matrix(y_true=y_test[:, 3], y_pred=prediction_floor)
        accuracy = (np.trace(cm_floor) / float(np.sum(cm_floor))) * 100
        subs = np.abs(np.array(prediction_floor, ndmin=2).T - np.array(y_test[:, 3], ndmin=2).T)
        true_false_values = np.array(subs, ndmin=2)
        return accuracy, true_false_values

    def building_hit_rate(self, X_test=None, y_test=None):
        prediction_building = self.classifier_building.predict(X_test)
        cm_building = confusion_matrix(y_true=y_test[:, 4], y_pred=prediction_building)
        accuracy = (np.trace(cm_building) / float(np.sum(cm_building))) * 100
        return accuracy
