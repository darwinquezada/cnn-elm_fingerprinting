import argparse
import os
import random
from miscellaneous.misc import Misc
from run_cnn_elm import run_cnn_elm
from run_elm import run_elm
from run_knn import run_knn
from numpy.random import seed, default_rng

# For reproducibility
rnd_seed = 11
random.seed(rnd_seed)
seed(rnd_seed)
default_rng(rnd_seed)


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
    p.add_argument('--config-file', dest='config_file', action='store', default='', help='Config file')
    p.add_argument('--dataset', dest='dataset', action='store', default='', help='Dataset')
    p.add_argument('--algorithm', dest='algorithm', action='store', default='', help='Algorithm')

    args = p.parse_args()

    # Check if the the config file exist
    config_file = str(args.config_file)
    datasets = str(args.dataset)
    algorithm = str(args.algorithm)

    misc = Misc()

    if config_file == '':
        print(misc.log_msg("ERROR", "Please specify the config file \n"
                                    " e.g., python main.py --config-file config.json \n"
                                    "or \n python main.py --config-file config.json --dataset DSI1,DSI2"))
        exit(-1)

    if os.path.exists(config_file):
        pass
    else:
        print(misc.log_msg("ERROR", "Oops... Configuration file not found. Please check the name and/or path."))
        exit(-1)

    # Config file from .json to dict
    config = misc.json_to_dict(config_file)

    # Check if all the parameters are present in the configuration file
    misc.conf_parse(config)

    # Get all the datasets availables in the config file
    list_datasets = misc.get_datasets_availables(config)

    if datasets != '':
        datasets = datasets.split(',')
        for i, dataset in enumerate(datasets):
            for j in range(0, len(config['dataset'])):
                if dataset == config['dataset'][j]['name']:
                    main_path = config['path']['data_source']
                    if os.path.exists(os.path.join(main_path, dataset,
                                                   config['dataset'][j]['train_dataset'])) and \
                            config['dataset'][j]['train_dataset'] != "":
                        pass
                    else:
                        print(misc.log_msg("ERROR",
                                           "Train Dataset not found."))
                        exit(-1)

                    if os.path.exists(os.path.join(main_path, dataset,
                                                   config['dataset'][j]['test_dataset'])) and \
                            config['dataset'][j]['test_dataset'] != "":
                        pass
                    else:
                        print(misc.log_msg("ERROR",
                                           "Test Dataset not found."))
                        exit(-1)

                    if os.path.exists(os.path.join(main_path, dataset,
                                                   config['dataset'][j]['validation_dataset'])):
                        pass
                    else:
                        print(misc.log_msg("ERROR",
                                           "Validation Dataset not found."))
                        exit(-1)

                    if algorithm == 'CNN-ELM':
                        run_cnn_elm(dataset_name=dataset, path_config=config['path'], dataset_config=config['dataset'][j],
                                    cnn_config=config['model_config'][0], elm_config=config['model_config'][1])
                    elif algorithm == "ELM":
                        run_elm(dataset_name=dataset, path_config=config['path'],dataset_config=config['dataset'][j],
                                elm_config=config['model_config'][1])
                    elif algorithm == "KNN":
                        run_knn(dataset_name=dataset, path_config=config['path'], dataset_config=config['dataset'][j],
                                knn_config=config['model_config'][2])
                    else:
                        print(misc.log_msg("ERROR",
                                           "Algorithm not available"
                                           "{CNN-ELM|ELM}"))
                        exit(-1)

    else:
        for dataset in config['dataset']:
            dataset_name = dataset['name']
            if algorithm == 'CNN-ELM':
                run_cnn_elm(dataset_name=dataset, path_config=config['path'], dataset_config=dataset,
                            cnn_config=config['model_config'][0], elm_config=config['model_config'][1])
            elif algorithm == 'ELM':
                run_elm(dataset_name=dataset, path_config=config['path'],
                        dataset_config=dataset, elm_config=config['model_config'][1])
            elif algorithm == "KNN":
                run_knn(dataset_name=dataset, path_config=config['path'], dataset_config=dataset,
                        knn_config=config['model_config'][2])
            else:
                print(misc.log_msg("ERROR",
                                   "Algorithm not available"
                                   "{CNN-ELM|ELM}"))
                exit(-1)