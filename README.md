<br />
<p align="center"> 
  <h3 align="center">Lightweight Hybrid CNN-ELM Model for Multi-building and Multi-floor Classification</h3>
</p>

[![pub package](https://img.shields.io/badge/license-CC%20By%204.0-green)]()

<!-- ABOUT THE PROJECT -->
## Abstract

Machine learning models have become an essential tool in current indoor positioning solutions, given their high capa-bilities to extract meaningful information from the environment. Convolutional neural networks (CNNs) are one of the most used neural networks (NNs) due to that they are capable of learning complex patterns from the input data. Another model used in indoor positioning solutions is the Extreme Learning Machine (ELM), which provides an acceptable generalization performance as well as a fast speed of learning. In this paper, we offer a lightweight combination of CNN and ELM, which provides a quick and accurate classification of building and floor, suitable for power and resource-constrained devices. As a result, the proposed model is 58% faster than the benchmark, with a slight improvement in the classification accuracy (by less than 1 %).


```
@INPROCEEDINGS{9797021,
  author={Quezada-Gaibor, Darwin and Torres-Sospedra, Joaquín and Nurmi, Jari and Koucheryavy, Yevgeni and Huerta, Joaquín},
  booktitle={2022 International Conference on Localization and GNSS (ICL-GNSS)}, 
  title={Lightweight Hybrid CNN-ELM Model for Multi-building and Multi-floor Classification}, 
  year={2022},
  volume={},
  number={},
  pages={01-06},
  doi={10.1109/ICL-GNSS54081.2022.9797021}}
```


### Developed using

* [Python](https://www.python.org/)


<!-- structure -->
## Getting Started

    .
    ├── datasets                      # WiFi/BLE fingerprinting datasets
    ├── miscellaneous                    
    │   ├── error_estimation.py       # Positioning error
    │   ├── misc.py                   # Miscellaneous functions
    │   └── plot_confusion_matrix.py  # Confusion matrix
    ├── model                    
    │   ├── cnn.py                   # CNN model
    │   ├── elm.py                   # ELM model
    │   └── knn.py                   # KNN algorithm
    ├── plots
    ├── positioning
    │   └── position.py               # KNN (Classifier and regressor)
    ├── preprocessing
    │   ├── data_processing.py        # Normalization, Standardization, ...
    │   └── data_representation.py    # Positive, Powerd, etc.
    ├── results
    ├── report  
    │   └── plot_results.py           # Results                    
    ├── savel_model                   # Generate plots
    ├── config.json                   # Configuration file
    ├── main.py                       # Main file
    ├── run_knn.py                    # Run KNN model
    ├── run_elm.py                    # Run ELM model
    ├── run_cnn_elm.py                # Run CNN-ELM model
    ├── requirements.txt              # Python libraries - requirements
    ├── license              
    └── README.md                     # The most important file :)

## Libraries
* pandas, numpy, seaborn, matplotlib, sklearn, colorama

## Datasets 
The datasets can be downloaded either from authors' repository (see README file in datasets folder) or from the following repository:

      "Joaquín Torres-Sospedra, Darwin Quezada-Gaibor, Germán Mendoza-Silva,
      Jari Nurmi, Yevgeny Koucheryavy, & Joaquín Huerta. (2020). Supplementary
      Materials for 'New Cluster Selection and Fine-grained Search for k-Means
      Clustering and Wi-Fi Fingerprinting' (1.0).
      Zenodo. https://doi.org/10.5281/zenodo.3751042"

## Converting datasets from .mat to .csv
1.- Copy the original datasets (.mat) into **dataset** folder.
2.- Modify the list of datasets in the /miscellaneous/datasets_mat_to_csv.py (line 23) with the dataset or datasets to be converted to csv.
```py
list_datasets = [ 'LIB1', 'LIB2', 'TUT1', 'TUT2', 'TUT3', 'TUT4', 'TUT5', 'TUT6', 'TUT7','UJI1','UTS1']
```

3.- Run the /miscellaneous/datasets_mat_to_csv.py.
```sh
  $ python /miscellaneous/datasets_mat_to_csv.py
```
## Usage
General parameters:
  * --config-file : Datasets and model's configuration (see config.json)
  * -- dataset : Dataset or datasets to be tested (i.e., UJI1 or UJI1,UJI2,TUT1)
  * -- algorithm : KNN, ELM or CNN-ELM.

* **Classification**
```sh
  $ python main.py --config-file config.json --dataset LIB1 --algorithm CNN-ELM
```
**_NOTE:_**
The hyperparameters of each model can be changed in the config file (config.json).

<!-- LICENSE -->
## License

CC By 4.0


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
The authors gratefully acknowledge funding from the European Union’s Horizon 2020 Research and Innovation programme under the Marie Sk\l{}odowska Curie grant agreement No. $813278$, A-WEAR.
