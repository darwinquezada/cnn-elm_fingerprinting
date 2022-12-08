import json
from colorama import init, Fore, Back, Style
import keras.backend as K
from keras.backend import elu, relu, abs, tanh, sigmoid, sin, cos
from tensorflow.python.keras.optimizers import adam_v2, adamax_v2, adadelta_v2, adagrad_v2, ftrl, nadam_v2, rmsprop_v2


class Misc:
    def json_to_dict(self, config_file):
        # Opening JSON file
        with open(config_file) as json_file:
            dictionary = json.load(json_file)
        return dictionary

    def check_key(self, dict, list_parameters):
        for param in range(0, len(list_parameters)):
            if list_parameters[param] in dict.keys():
                pass
            else:
                print(self.log_msg("ERROR", " The following parameter is not found in the configuration file: " +
                                   list_parameters[param]))
                exit(-1)
        return True

    def log_msg(self, level, message):
        init(autoreset=True)
        if level == 'WARNING':
            return Fore.YELLOW + message
        elif level == 'ERROR':
            return Fore.RED + message
        elif level == 'INFO':
            return Style.RESET_ALL + message

    def conf_parse(self, dict):
        # These parameters are compulsory in the config file
        conf_main_param = ['path', 'dataset', 'model_config']
        dataset_param = ['name', 'data_representation', 'default_null_value', 'train_dataset', 'test_dataset',
                         'validation_dataset']
        model_elm = ['model', 'train', 'hidden_neurons', 'act_funct']
        model_cnn = ['model', 'type', 'train', 'padding', 'strides', 'data_format', 'act_funct', 'kernel_size',
                     'filter']
        model_dnn = ["model", "train", "lr", "batch_size", "epochs", "loss", "optimizer"]
        model_knn = ["model", "train"]

        # Check if all the main parameters are in the config file
        if self.check_key(dict, conf_main_param):
            pass

        # Datasets parameters
        for data in dict['dataset']:
            if self.check_key(data, dataset_param):
                pass

        # Models' parameters
        for data in dict['model_config']:
            if data['model'] == 'ELM':
                if self.check_key(data, model_elm):
                    pass
            elif data['model'] == 'DNN':
                if self.check_key(data, model_dnn):
                    pass
            elif data['model'] == 'STACKED':
                if self.check_key(data, model_dnn):
                    pass
            elif data['model'] == 'KNN':
                if self.check_key(data, model_knn):
                    pass
            else:
                if self.check_key(data, model_cnn):
                    pass

    def get_datasets_availables(self, dict):
        list_datasets = []
        for data in dict['dataset']:
            list_datasets.append(data['name'])
        return list_datasets

    def activation_function(self, x, function):
        if function == "tansig":
            r = 2 / (1 + K.exp(-2 * x)) - 1
        elif function == "tanh":
            r = tanh(x)
        elif function == "linsat":
            r = K.abs(1 + x) - K.abs(1 - x)
        elif function == "relu":
            r = relu(x)
        elif function == "elu":
            r = elu(x)
        elif function == "sigmoid":
            r = sigmoid(x)
        elif function == "abs":
            r = abs(x)
        elif function == "sine":
            r = sin(x)
        elif function == "cosine":
            r = cos(x)
        elif function == "linear":
            r = x
        else:
            print(self.log_msg("ERROR", " Activation function not valid."))
            exit(-1)
        return r

    def optimizer(self, opt, lr):
        if opt == 'Adam':
            return adam_v2.Adam(lr)
        elif opt == 'Adamax':
            return adamax_v2.Adamax(lr)
        elif opt == 'Adadelta':
            return adadelta_v2.Adadelta(lr)
        elif opt == 'Adagrad':
            return adagrad_v2.Adagrad(lr)
        elif opt == 'Ftrl':
            return ftrl.Ftrl(lr)
        elif opt == 'Nadam':
            return nadam_v2.Nadam(lr)
        elif opt == 'RMSprop':
            return rmsprop_v2.RMSprop(lr)
        else:
            return adam_v2.Adam(lr)
