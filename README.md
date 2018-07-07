# DataTools

This package is meant to hold various tools that might be useful for analyzing data and running machine learning models.

Currently the main focus is allowing for faster prototyping of models.

Some basic features that I've incorporated into this package include:

  - Basic dataset and result classes
  - Batch generation
  - K-fold CV with out-of-fold predictions and test predictions for each fold
  - Tensorflow training for fairly simple models
  - Tensorflow feed-dict production
  
 I also expect to have some basic tools for other packages such as LightGBM, Keras, and XGBoost.
 
 Some things still to do include:
  
   - Add model serialization to the train & predict functions
   - Deserialization & prediction with no training
   - Basic model architectures
   - Embedding dataset production