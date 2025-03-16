**Project Overview**

This class project applies machine learning to drum sound classification, involving feature extraction, data preprocessing, and training a convolutional neural network.
The dataset consists of kicks, snares, cymbals, and toms collected from various free online sources, spanning a range from acoustic to heavily processed metal. The full dataset is not included in this repository, but instructions for obtaining it, along with additional details, are available in the included final report.

**Usage**

Run individual_cleaning.py to process and categorize drum samples into their respective folders.
After initial preprocessing, run make_total.py to consolidate and shuffle the samples, ensuring a balanced dataset.
Run drum_sound_classification.py, which calls drum_sample_processing.py internally. Together they:
- Perform final processing
- Split the dataset into training/validation/testing sets
- Train the model
- Generate evaluation plots
For further details on methodology and results, refer to the included report.

**Requirements**

This project was developed using Python. Install the required dependencies before running the scripts:
- numpy
- pandas
- librosa
- torch
- sklearn
- matplotlib
- pillow
- soundfile






**Note**

The stopwatch module belongs to my advisor Robert Nickel.
