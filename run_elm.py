from preprocessing.data_preprocessing import load, new_non_detected_value
from preprocessing.data_representation import DataRepresentation
from sklearn.preprocessing import Normalizer, LabelEncoder, OneHotEncoder
from miscellaneous.plot_confusion_matrix import plot_confusion_matrix
from model.elm import elmTrain_fix, elmPredict_optim
# from skelm.elm import ELMRegressor, ELMClassifier
from sklearn.metrics import confusion_matrix
from miscellaneous.misc import Misc
import keras.backend as K
from datetime import datetime
import time as ti
import joblib
import logging
import os
import numpy as np

'''
Based on:
R. Dogaru and I. Dogaru, "BCONV - ELM: Binary Weights Convolutional Neural Network 
Simulator based on Keras/Tensorflow, for Low Complexity Implementations," 
2019 6th International Symposium on Electrical and Electronics Engineering 
(ISEEE), 2019, pp. 1-6, doi: 10.1109/ISEEE48094.2019.9136102.
'''

def run_elm(dataset_name=None, path_config=None, dataset_config=None, elm_config=None):

    misc = Misc()
    dataset_path = os.path.join(path_config['data_source'], dataset_name)
    main_path_save = os.path.join(path_config['saved_model'], dataset_config['name'], elm_config['model'])

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
    if elm_config['train']:
        data_norm_path = os.path.join(main_path_save, "data_norm")
        norm = Normalizer()
        X_train = norm.fit_transform(X_train)
        X_test = norm.transform(X_test)

        if not os.path.exists(data_norm_path):
            os.makedirs(data_norm_path)

        joblib.dump(norm, data_norm_path + '/data_normalization.save')

        # Data reshape
        x_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        x_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Foor labels
        y_train_floor = y_train.iloc[:, 3].values
        y_test_floor = y_test.iloc[:, 3].values

        # Building labels
        y_train_bld = y_train.iloc[:, 4].values
        y_test_bld = y_test.iloc[:, 4].values

        # Label encoding
        lab_enc = LabelEncoder()
        joined_lab_enc = np.concatenate((y_train_floor, y_test_floor), axis=0)
        y_lab_enc = lab_enc.fit(joined_lab_enc)
        y_train_floor = lab_enc.transform(y_train_floor)
        y_test_floor = lab_enc.transform(y_test_floor)

        # Label encoding Building
        lab_enc_bld = LabelEncoder()
        y_train_bld = lab_enc_bld.fit_transform(y_train_bld)
        y_test_bld = lab_enc_bld.transform(y_test_bld)

        # Onehot encoding
        encoder = OneHotEncoder(sparse=False)
        y_train_floor_oe = encoder.fit_transform(y_train_floor.reshape(-1, 1))
        y_test_floor_oe = encoder.fit_transform(y_test_floor.reshape(-1, 1))

        encoder_bld = OneHotEncoder(sparse=False)
        y_train_bld_oe = encoder_bld.fit_transform(y_train_bld.reshape(-1, 1))
        y_test_bld_oe = encoder_bld.fit_transform(y_test_bld.reshape(-1, 1))

        # Permutation
        idx = np.random.permutation(len(x_train))
        x_train = x_train[idx]
        y_train_floor_oe = y_train_floor_oe[idx]
        y_train_build_oe = y_train_bld_oe[idx]

        # Training Models

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # ELM
    intrain = K.variable(x_train)
    Samples_ = K.batch_flatten(intrain)  # aici se aplica direct datele de intrare
    SamplesTrain = (K.eval(Samples_)).T

    Labels = y_train_floor_oe
    Labels_bld = y_train_build_oe

    # ELM - training
    t1 = ti.time()
    inW, outW, *_ = elmTrain_fix(SamplesTrain, np.transpose(Labels), elm_config["hidden_neurons"], elm_config["C"],
                                 elm_config["act_funct"], elm_config["win_bits"])
    inW_bld, outW_bld, *_ = elmTrain_fix(SamplesTrain, np.transpose(Labels_bld), elm_config["hidden_neurons"],
                                         elm_config["C"], elm_config["act_funct"], elm_config["win_bits"])
    ttrain = ti.time() - t1
    print(" training time: %f seconds" % ttrain)

    # ==============  Quantify the output layer ======================================
    Qout = -1 + pow(2, elm_config["wout_bits"] - 1)
    if elm_config["wout_bits"] > 0:
        O = np.max(np.abs(outW))
        outW = np.round(outW * (1 / O) * Qout)

        O_bld = np.max(np.abs(outW_bld))
        outW_bld = np.round(outW_bld * (1 / O_bld) * Qout)

    # ================= TEST (VALIDATION) DATASET LOADING
    intest = K.variable(x_test)
    Samples_ = K.batch_flatten(intest)  # aici se aplica direct datele de intrare
    SamplesTest = (K.eval(Samples_)).T

    # ====================== VALIDATION PHASE (+ Accuracy evaluation) =================
    t2 = ti.time()
    scores, *_ = elmPredict_optim(SamplesTest, inW, outW, elm_config["act_funct"])
    scores_bld, *_ = elmPredict_optim(SamplesTest, inW_bld, outW_bld, elm_config["act_funct"])
    ttest = ti.time() - t2
    print(" prediction time: %f seconds" % ttest)

    round_predictions = np.argmax(np.transpose(scores), axis=-1)
    cm = confusion_matrix(y_true=y_test_floor, y_pred=round_predictions)
    accuracy = (np.trace(cm) / float(np.sum(cm))) * 100

    # Building
    round_predictions_bld = np.argmax(np.transpose(scores_bld), axis=-1)
    cm_bld = confusion_matrix(y_true=y_test_bld, y_pred=round_predictions_bld)
    accuracy_bld = (np.trace(cm_bld) / float(np.sum(cm_bld))) * 100

    labels = np.unique(y_train_floor)

    datestr = "%m/%d/%Y %I:%M:%S %p"
    save_path_log = os.path.join("results", str(dataset_name), elm_config['model'], "LOG")

    if not os.path.exists(save_path_log):
        os.makedirs(save_path_log)

    logging.basicConfig(
        filename=save_path_log + '/' + datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + '.log',
        level=logging.INFO,
        filemode="w",
        datefmt=datestr,
        # force=True
    )

    print("Confusion matrix is: ")
    print("Floor hit rate: %f" % accuracy)
    print("Building hit rate: %f" % accuracy_bld)
    print("Number of hidden neurons: %d" % elm_config["hidden_neurons"])
    print("Activation function:" + elm_config["act_funct"])

    logging.info("---------------------------- ELM ---------------------------")
    logging.info(' Dataset : ' + dataset_name)
    logging.info(' Training time : {:.6f}'.format(ttrain))
    logging.info(' Prediction time : {:.6f}'.format(ttest))
    logging.info(' Floor hit rate : {:.3f}'.format(accuracy))
    logging.info(' Building hit rate : {:.3f}'.format(accuracy_bld))
    logging.info(' ------- ELM configuration ------- ')
    logging.info(" model: " + elm_config["model"])
    logging.info(" train: " + str(elm_config["train"]))
    logging.info(" hidden_neurons: " + str(elm_config["hidden_neurons"]))
    logging.info(" act_funct: " + elm_config["act_funct"])
    logging.info(" C: " + str(elm_config["C"]))
    logging.info(" win_method: " + elm_config["win_method"])
    logging.info(" win_bits: " + str(elm_config["win_bits"]))
    logging.info(" wout_bits: " + str(elm_config["wout_bits"]))

    K.clear_session()

    return True