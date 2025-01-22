# lec-anonymization-som
Datasets for Local Energy Communities and code for k-anonymization and training/visualization of Self-Organizing maps

This repository includes the source code and input/output datasets for the research article entitled 
"Data anonymization and aggregation approaches for local energy communities", 
submitted for publication in ACM Energy Informatics Review.
For transparency, reproducibility and experimentation, all datasets that were used/exported in the paper are included, 
accompanied with the python source code. 

The application includes two functionalities, a Flask app exposing an anonymized dataset in JSON format, using an 
implementation of k-anonymization Mondrian multidimensional partitioning, with suppression and generalization for datasets, and 
a data aggregation approach with training of Self-organizing maps for visualizing datasets. MiniSom open source software package is 
used for training SOMs. 

Two main scenarios are included: A local energy community with 60 metering devices and energy-related variables for k-anonymization, 
accompanied with SOM training of a LEC generated dataset for 30 days, including energy-related variables and sensor-data, and 
training of a SOM for one metering device, i.e. residential scenario, for 30 days.

For visualization purposes, multiple figures are exported for each trained SOM, with clustering of nodes, for intensity maps, heatmaps, 
U-matrices.

Insights and parts of this code have been supported by LLMs.

The source code is provided under an Apache License, version 2.0.
