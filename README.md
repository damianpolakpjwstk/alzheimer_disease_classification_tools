# alzheimer_disease_classification_tools

This project contains the code to analyze and classify MRI scans to predict the Alzheimer's disease and
Mild Cognitive Impairment (MCI) progression. It uses two datasets: ADNI and BIOCARD (see below: Scans preparation).
It uses 3D convolutional neural networks (CNN) to classify the scans. Also, it uses XGBoost and SVM to classify
volumetric features extracted from the scans using FreeSurfer. Repository contains the whole process - from the scans
preparation to the classification and the result analysis.

Repository consists of the following folders, each containing the scripts to perform steps of the analysis:

* `scans_preparation` - scripts to prepare the scans for CNN training and testing (see below: Scans preparation)
* `data_analysis_preparation` - scripts to prepare and analyze the data (see below: Data analysis and preparation)
* `tabular_models` - scripts to train and test the tabular models using extracted volumetric data (see below: Tabular models)
* `imaging_models` - scripts to train and test the CNN models (see below: CNN models)

### Data acquisition

To reproduce the results, you need to download the ADNI and BIOCARD datasets. The ADNI dataset is available at
http://adni.loni.usc.edu/. The BIOCARD dataset is available at https://www.biocard-se.org. Both datasets are
not publicly available and require a registration and a data use agreement. See docstring in the 
`data_analysis_preparation` and `scans_preparation` directories for more details (which files to download and where to put them).

### Scans preparation

This folder contains the scripts used to prepare the scans for CNN training and testing. After the scans are downloaded,
they need to be preprocessed. The preprocessing includes the following steps:

* Skull stripping with HD-BET (https://github.com/MIC-DKFZ/HD-BET), (https://doi.org/10.1002/hbm.24750)
* Bias field correction with N4ITK using ANTs (https://pubmed.ncbi.nlm.nih.gov/20378467/), (http://stnava.github.io/ANTs/)
* Registration to MNI152 template with FLIRT (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT)
* Denoising with median filter
* Intensity rescaling
* Histogram equalization

This preprocessing is performed with the `run_adni_scans_preprocessing.py` and `run_biocard_scans_preprocessing.py`
scripts. For ADNI dataset, the preprocessing is performed on any subset of study, depending on found scans.

You need to perform this step first, before running any other scripts. This is very time-consuming step.

### Data analysis and preparation

This folder contains the scripts used to prepare and analyze the data. The data is prepared for the tabular models
and for the CNN models. It contains statistical analysis of the data (in notebooks) and the data visualization. To
perform next steps, you need to run `adni_train_val_test_split.py` and `biocard_generate_test_set.py` scripts. These 
scripts generate the train, validation and test sets for the tabular models and the CNN models as .csv files.

### Tabular models

The tabular models use the volumetric data (only on the ADNI dataset). The volumetric data extracted with FreeSurfer are
provided by ADNI. `ADNI_tabular_models.ipynb` notebook contains the code to train and test the XGBoost and SVM models.
This notebook also contains the code to extract the most important features from the models to explain the predictions.

This step is optional.

### CNN models

The CNN models uses the preprocessed scans. The CNN models are trained and tested with the
`imaging_models/CNN_training.ipynb` notebook. Additional evaluation of the models is performed with the
scans from the BIOCARD dataset. The BIOCARD dataset is not used for training, only for testing, as independent test set.
It uses my implementation of the Pseudo-3D CNN models (https://github.com/damianpolakpjwstk/pseudo3d_pytorch) with
attention blocks. See the notebook, `imaging_models/model.py` and README in pseudo3d_pytorch repository for more details.

TODO: add explanation of the CNN models with Grad-CAM visualization.

### Installation

Repository uses Python 3.10. 

To use the code, you need to install the following tools:

* FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation)
* ANTs (http://stnava.github.io/ANTs/)
* HD-BET (https://github.com/MIC-DKFZ/HD-BET)
* Python packages from `requirements.txt` file

After installing the tools, you need to set the paths to the tools in `scans_preparation/configuration.py` file.

