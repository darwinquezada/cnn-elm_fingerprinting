from preprocessing.data_preprocessing import load, new_non_detected_value
from preprocessing.data_representation import DataRepresentation
from sklearn.metrics import confusion_matrix
from miscellaneous.misc import Misc
# from skelm.elm import ELMRegressor
from model.cnn import convlayer
from sklearn.preprocessing import Normalizer, MinMaxScaler, LabelEncoder, OneHotEncoder, RobustScaler
from miscellaneous.plot_confusion_matrix import plot_confusion_matrix
import keras.backend as K
from datetime import datetime
import scipy.io as sci
import time as ti
import joblib
import os
import logging
import numpy as np
from model.elm import elmTrain_fix, elmPredict_optim

'''
Based on:
R. Dogaru and I. Dogaru, "BCONV - ELM: Binary Weights Convolutional Neural Network 
Simulator based on Keras/Tensorflow, for Low Complexity Implementations," 
2019 6th International Symposium on Electrical and Electronics Engineering 
(ISEEE), 2019, pp. 1-6, doi: 10.1109/ISEEE48094.2019.9136102.
'''

def run_cnn_elm(dataset_name=None, path_config=None, dataset_config=None, cnn_config=None, elm_config=None):

    misc = Misc()
    dataset_path = os.path.join(path_config['data_source'], dataset_name)
    main_path_save = os.path.join(path_config['saved_model'], dataset_config['name'], cnn_config['model']+'-'+elm_config['model'])

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

    if cnn_config['train'] == True or elm_config['train'] == True:
        data_norm_path = os.path.join(main_path_save, "data_norm")

        # Normalize
        norm = Normalizer()
        X_train = norm.fit_transform(X_train)
        X_test = norm.transform(X_test)

        if not os.path.exists(data_norm_path):
            os.makedirs(data_norm_path)

        joblib.dump(norm, data_norm_path + '/data_normalization.save')

        # Data reshape
        x_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        x_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Floor labels
        y_train_floor = y_train.iloc[:, 3].values
        y_test_floor = y_test.iloc[:, 3].values

        # Building labels
        y_train_bld = y_train.iloc[:, 4].values
        y_test_bld = y_test.iloc[:, 4].values

        # Label encoding Floor
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
    # General
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    intrain = K.variable(x_train)
    intest = K.variable(x_test)

    # CNN Model
    t1 = ti.time()
    out_train_ = convlayer(intrain, cnn_config=cnn_config)
    out_test_ = convlayer(intest, cnn_config=cnn_config)

    # Effective computation of the input preprocessing flow

    out_train = K.eval(out_train_)
    out_test = K.eval(out_test_)
    ttrain_cnn = ti.time()-t1
    K.clear_session()  # to avoid overloads

    # ELM
    Samples = out_train.T
    Labels_floor = y_train_floor_oe
    Labels_bld = y_train_build_oe

    # ELM - training
    t1 = ti.time()
    inW, outW, h_train = elmTrain_fix(Samples, np.transpose(Labels_floor), elm_config["hidden_neurons"],
                                      elm_config["C"], elm_config["act_funct"], elm_config["win_bits"])

    inW_bld, outW_bld, h_train_bld = elmTrain_fix(Samples, np.transpose(Labels_bld), elm_config["hidden_neurons"],
                                      elm_config["C"], elm_config["act_funct"], elm_config["win_bits"])
    ttrain = ti.time() - t1 + ttrain_cnn

    print(" training time: %f seconds" % ttrain)
    # ==============  Quantify the output layer ======================================
    Qout = -1 + pow(2, elm_config["wout_bits"] - 1)
    if elm_config["wout_bits"] > 0:
        O = np.max(np.abs(outW))
        outW = np.round(outW * (1 / O) * Qout)

        O_bld = np.max(np.abs(outW_bld))
        outW_bld = np.round(outW_bld * (1 / O_bld) * Qout)

    # ================= TEST (VALIDATION) DATASET LOADING

    SamplesTest = out_test.T

    # ====================== VALIDATION PHASE (+ Accuracy evaluation) =================
    t2 = ti.time()
    scores, h_test = elmPredict_optim(SamplesTest, inW, outW, elm_config["act_funct"])
    scores_bld, h_test_bld = elmPredict_optim(SamplesTest, inW_bld, outW_bld, elm_config["act_funct"])

    ttest = ti.time() - t2
    print(" prediction time: %f seconds" % ttest)

    # Floor prediction
    round_predictions = np.argmax(np.transpose(scores), axis=-1)
    cm = confusion_matrix(y_true=y_test_floor, y_pred=round_predictions)
    accuracy = (np.trace(cm) / float(np.sum(cm))) * 100

    # Building
    round_predictions_bld = np.argmax(np.transpose(scores_bld), axis=-1)
    cm_bld = confusion_matrix(y_true=y_test_bld, y_pred=round_predictions_bld)
    accuracy_bld = (np.trace(cm_bld) / float(np.sum(cm_bld))) * 100

    labels = np.unique(y_train_floor)

    # ----------------------------------- OS-ELM -------------------------------------
    # oselmc = OSELMClassifier(n_hidden=np.shape(np.transpose(Samples))[1], activation_func='hardlim', random_state=1102)
    # oselmc.fit(np.transpose(Samples), Labels)
    # print("Test score of total: %s" % str(oselmc.score(np.transpose(SamplesTest), y_test_floor_oe)*100))

    # ------------------------------------- SVM --------------------------------------

    # svm_classifier = svm.SVC()
    # t1 = ti.time()
    # svm_classifier.fit(out_train, y_train_floor)
    # ttrain = ti.time() - t1
    # t2 = ti.time()
    # prediction = svm_classifier.predict(out_test)
    # tpred = ti.time() - t2
    # acc = accuracy_score(y_test_floor, prediction)
    # print("------------------------ SVM ---------------------")
    # print("Training time: {:.6f}".format(ttrain))
    # print("Prediction time: {:.6f}".format(tpred))
    # print("Accuracy SVM: {:.3f}".format(acc * 100))
    # print("-------------------------------------------------")

    '''
    skelm_regressor = ELMRegressor(alpha=1e-3, n_neurons=530, pairwise_metric='euclidean', batch_size=40)
    skelm_regressor.fit(np.transpose(Samples), y_train.iloc[:, 0:3])
    pred = skelm_regressor.predict(np.transpose(SamplesTest))
    errors = pred[:, 0:3] - y_test.iloc[:, 0:3].values
    distances = np.linalg.norm(errors, ord=2, axis=1)
    print(np.mean(distances))
    '''
    
    # ------------------------------------ Report ------------------------------------

    datestr = "%m/%d/%Y %I:%M:%S %p"
    save_path_log = os.path.join("results", str(dataset_name), cnn_config['model']+'-'+elm_config['model'], "LOG")

    if not os.path.exists(save_path_log):
        os.makedirs(save_path_log)

    logging.basicConfig(
        filename=save_path_log + '/'+datetime.today().strftime('%Y_%m_%d_%H_%M_%S')+'.log',
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

    logging.info("---------------------------- CNN-ELM ---------------------------")
    logging.info(' Dataset : ' + dataset_name)
    logging.info(' Training time : {:.6f}'.format(ttrain))
    logging.info(' Prediction time : {:.6f}'.format(ttest))
    logging.info(' Floor hit rate : {:.2f}'.format(accuracy))
    logging.info(' Building hit rate : {:.2f}'.format(accuracy_bld))
    logging.info(' ------- CNN configuration ------- ')
    logging.info(" model: " + cnn_config['model']+'-'+elm_config['model'])
    logging.info(" type: " + str(cnn_config["type"]))
    logging.info(" train: " + str(cnn_config["train"]))
    logging.info(" padding: " + cnn_config["padding"])
    logging.info(" strides: " + str(cnn_config["strides"]))
    logging.info(" data_format: " + cnn_config["data_format"])
    logging.info(" act_funct: " + cnn_config["act_funct"])
    logging.info(" kernel_size: " + str(cnn_config["kernel_size"]))
    logging.info(" filter: " + str(cnn_config["filter"]))
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

    # plot_confusion_matrix(cm=cm, normalize=False, target_names=labels,
    #                       title=(dataset_name + " - Alg.: CNN-ELM"),
    #                       title_figure=(dataset_name + "_CM_CNN-ELM"), dataset=dataset_name)

    return True