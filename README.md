# MACHINE LEARNING - PROJECT 1
## HIGGS BOSON CHALLENGE  
---------------------------------------------
Authors: Riccardo Succa, Aleksandr Tukallo, Marco Zoveralli
_____________________________________________
### The Dataset
The dataset consists in 250'000 samples, each with 30 features, of particle collisions conducted ad CERN (Geneva).

### Project Goal
This project aims to use machine learning algorithms to predict the decay signature of particle collisions and understand if the event’s signature was the result of a Higgs boson (signal) or some other process/particle (background).
No machine learning libraries have been used.

### Running the Model
The model can be trained by downloading the dataset (and put it inside a data/ folder) and running the python script `run.py`. Moreover, the Jupyter notebook `sample_notebook.ipynb` describes in details the data analisys and the models tried.

### Contents
* `MLscripts/`: folder containing the python scripts used to load, clean, plot the dataset and the implementation of the baselines. 
* `hotgrad/`: Neural Network framework. 
* `sample_solution.ipynb`: jupyter notebook containing all the steps followed, from the data analysis to the tuning and training of the models. 
