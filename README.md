# Feature-Based Classification of Time-Series Sensor Signals

## Overview  
This project extends basic signal processing into feature extraction and classification. An Arduino is used to generate controlled optical signals using an LED and a photoresistor. The resulting time-series data is processed in Python to extract spectral features, which are then used to train a supervised machine learning model that can automatically distinguish between different signal conditions.

The goal is to move beyond visual inspection of FFT plots and build a quantitative, feature-driven pipeline for classifying noisy sensor data.

---

## What This Project Does  

- Collects timestamped sensor data from an Arduino over serial  
- Generates controlled optical signals (LED off, 10Hz flicker, 20Hz flicker)  
- Removes DC offset and converts signals to the frequency domain using FFT  
- Extracts interpretable spectral features from each recording  
- Trains a supervised classifier to distinguish between signal conditions  
- Evaluates model performance using accuracy and a confusion matrix  

---

## Hardware Setup  

- Arduino Uno  
- Photoresistor connected to A0  
- LED connected to digital pin 6  
- Resistors  
- Breadboard and jumper wires  

The LED is positioned near the photoresistor and flickered at controlled frequencies to inject known optical signals into the sensor data.

A TinkerCAD model of the hardware setup is included in the `figures/` folder.

---

## Software Pipeline  

### Arduino (`arduino/`)
- Controls LED flickering (off, 10Hz, 20Hz)  
- Reads photoresistor values  
- Streams data as `timestamp, ADC_value` over serial at 115200 baud  

### Analysis & Feature Extraction (`analysis and feature extraction/`)
- Reads serial data into Python  
- Calculates true sampling rate from timestamp differences  
- Removes DC offset  
- Performs FFT using `np.fft.rfft`  
- Extracts spectral features including:  
  - Dominant Frequency  
  - Band Power Ratio  
  - Peak Magnitude  
  - Spectral Centroid  
  - Spectral Spread  

Each 10-second recording is reduced to a compact feature vector and saved to `features.csv`.

### Data (`data/`)
- Stores `features.csv`, the dataset used for training the classifier  

### Classification (`ml_classification/`)
- Loads extracted features  
- Normalizes feature values  
- Splits data into training and testing sets  
- Trains a Logistic Regression classifier  
- Evaluates performance using accuracy and a confusion matrix  

---

## Results  

The classifier is able to reliably distinguish between:

- LED Off  
- 10Hz Flicker  
- 20Hz Flicker  

Using only spectral features, the model achieves high accuracy, demonstrating that the frequency-domain structure of the signal contains enough information for clean separation. Feature importance analysis shows that dominant frequency and peak magnitude are the strongest predictors.

---

## Repository Structure  

arduino/
Arduino_flickering.ino

analysis and feature extraction/
data_acquisition.py

ml_classification/
model_training.py

data/
features.csv

figures/
Arduino_TinkerCAD_Model.png


---

## Why This Matters  

This project mirrors how real biological and physical signals are handled in practice. Raw sensor data is rarely clean, and useful information must be extracted through signal processing before any machine learning can be applied. The same pipeline structure used here: acquisition, signal proccesing, spectral analysis, feature engineering, and classification.
