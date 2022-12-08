from preprocessing.data_preprocessing import load, new_non_detected_value
from preprocessing.data_representation import DataRepresentation
from sklearn.preprocessing import Normalizer, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from miscellaneous.misc import Misc
from datetime import datetime
import joblib
import time as ti
import logging
import os
import numpy as np

def run_knn(dataset_name=None, path_config=None, dataset_config=None, knn_config=None):

    misc = Misc()
    dataset_path = os.path.join(path_config['data_source'], dataset_name)
    main_path_save = os.path.join(path_config['saved_model'], dataset_config['name'], knn_config['model'])

    if bool(dataset_config['train_dataset']):
        X_train, y_train = load(os.path.join(dataset_path, dataset_config['train_dataset']))

    if bool(dataset_config['test_dataset']):
        X_test, y_test = load(os.path.join(dataset_path, dataset_config['test_dataset']))

    if bool(dataset_config['validation_dataset']):
        X_valid, y_valid = load(os.path.join(dataset_path, dataset_config['validation_dataset']))
    else:
        X_valid = []
        y_valid = []

    # Change data representation
    new_non_det_val = new_non_detected_value(X_train, X_test, X_valid)
    dr = DataRepresentation(x_train=X_train, x_test=X_test, x_valid=X_valid,
                            type_rep=dataset_config['data_representation'],
                            def_no_val=dataset_config['default_null_value'],
                            new_no_val=new_non_det_val)
    X_train, X_test, X_valid = dr.data_rep()

    # Normalize
    if knn_config['train']:
        data_norm_path = os.path.join(main_path_save, "data_norm")
        norm = Normalizer()
        X_train = norm.fit_transform(X_train)
        X_test = norm.transform(X_test)

        if not os.path.exists(data_norm_path):
            os.makedirs(data_norm_path)

        joblib.dump(norm, data_norm_path + '/data_normalization.save')
 
        # Foor labels
        y_train_floor = y_train.iloc[:, 3].values
        y_test_floor = y_test.iloc[:, 3].values

        # Label encoding
        lab_enc = LabelEncoder()
        joined_lab_enc = np.concatenate((y_train_floor, y_test_floor), axis=0)
        y_lab_enc = lab_enc.fit(joined_lab_enc)
        y_train_floor = lab_enc.transform(y_train_floor)
        y_test_floor = lab_enc.transform(y_test_floor)

        # Building labels
        y_train_build = y_train.iloc[:, 4].values
        y_test_build = y_test.iloc[:, 4].values

        # Training Models

    y_tr_fl = np.array(y_train_floor, ndmin=2)
    y_tr_bl = np.array(y_train_build, ndmin=2)

    # KNN - training
    t1 = ti.time()
    knn_classifier_floor = KNeighborsClassifier(n_neighbors=dataset_config['k'], metric=dataset_config['distance_metric'])
    knn_classifier_floor.fit(X_train, y_tr_fl.T)

    knn_classifier_bld = KNeighborsClassifier(n_neighbors=dataset_config['k'], metric=dataset_config['distance_metric'])
    knn_classifier_bld.fit(X_train, y_tr_bl.T)
    ttrain = ti.time()-t1

    t2 = ti.time()
    prediction_floor = knn_classifier_floor.predict(X_test)
    prediction_bld = knn_classifier_bld.predict(X_test)

    ttest = ti.time() - t2

    cm_floor = confusion_matrix(y_true=y_test_floor, y_pred=prediction_floor)
    cm_bld = confusion_matrix(y_true=y_test_build, y_pred=prediction_bld)
    accuracy_floor = (np.trace(cm_floor) / float(np.sum(cm_floor))) * 100
    accuracy_bld = (np.trace(cm_bld) / float(np.sum(cm_bld))) * 100

    # labels = np.unique(y_train_floor)

    datestr = "%m/%d/%Y %I:%M:%S %p"
    save_path_log = os.path.join("results", str(dataset_name), knn_config['model'], "LOG")

    if not os.path.exists(save_path_log):
        os.makedirs(save_path_log)

    logging.basicConfig(
        filename=save_path_log + '/' + datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + '.log',
        level=logging.INFO,
        filemode="w",
        datefmt=datestr,
        # force=True
    )

    print(' Building hit rate : {:.3f}'.format(accuracy_bld))
    print(' Floor hit rate : {:.3f}'.format(accuracy_floor))
    print(' Training time : {:.6f}'.format(ttrain))
    print(' Prediction time : {:.6f}'.format(ttest))
    print(' Full time : {:.6f}'.format(ttest+ttrain))

    logging.info("---------------------------- KNN ---------------------------")
    logging.info(' Dataset : ' + dataset_name)
    logging.info(' Building hit rate : {:.3f}'.format(accuracy_bld))
    logging.info(' Floor hit rate : {:.3f}'.format(accuracy_floor))
    logging.info(' Training time : {:.6f}'.format(ttrain))
    logging.info(' Prediction time : {:.6f}'.format(ttest))
    logging.info(' Full time : {:.6f}'.format(ttest+ttrain))
    logging.info(' ------- KNN configuration ------- ')
    logging.info(" model: " + knn_config["model"])
    logging.info(" K:" + str(dataset_config['k']))
    logging.info(" distance: " + dataset_config['distance_metric'])

    # plot_confusion_matrix(cm=cm, normalize=False, target_names=labels,
    #                       title=(dataset_name + " - Alg.: KNN"),
    #                       title_figure=(dataset_name + "_KNN"), dataset=dataset_name)

    return True