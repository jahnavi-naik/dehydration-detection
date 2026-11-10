# Personalized Dehydration Detection from Wearable Signals

## Project Overview

This project investigates whether physiological signals collected from wearable sensors can be used to detect changes associated with dehydration. We aim to develop and compare general population-level machine learning models with personalized models that incorporate individual physiological baselines.

Our primary research questions are:
- Can wearable signals such as PPG, GSR, temperature, and motion help identify dehydration?
- Do personalized models outperform general population models?
- Which physiological signals are most informative?
- Can we distinguish dehydration-related changes from other physiological responses, such as stress, exercise, or fasting?

## Dataset

**Source:** [Shimmer Raw Dataset for Dehydration — Zenodo](https://zenodo.org/records/6299964)

The primary dataset contains physiological recordings collected using Shimmer3 wearable sensors.

**Dataset characteristics:**
- 11 participants (9 female, 2 male)
- Approximately 3,386 minutes of recordings
- Approximately 1.05 GB compressed ZIP archive
- PPG and GSR sampled at 128 Hz; IMU sampled at 512 Hz, according to the project proposal
- Includes physiological measurements collected under different experimental conditions, including fasting

**Known limitations:**
- One participant (S1) contributes approximately 52% of the total recording duration.
- The participant population is small and demographically imbalanced.
- Fasting does not necessarily indicate dehydration.
- The availability and validity of dehydration ground-truth labels still need to be investigated.

## Data Exploration and Preprocessing

### 1. Dataset Acquisition and Inspection

- Downloaded and extracted the raw dataset from Zenodo.
- Used Python's `pathlib` library to navigate the extracted directory.
- Identified CSV files containing wearable sensor recordings.
- Loaded a sample of the data using pandas to inspect its structure.

### 2. Understanding the Raw Data Format

The Shimmer CSV files contain multiple metadata rows before the actual measurements:

1. Device identifier
2. Sensor/channel name
3. Calibration status (`CAL` or `UNCAL`)
4. Measurement units
5. Numerical sensor readings

Initial CSV inspection showed 38 columns, including duplicate calibrated and uncalibrated channels.

We identified the metadata structure to support correct column selection and preprocessing.

### 3. Sensor Selection

The following calibrated channels were identified for further analysis:

| Signal | Description | Original column index |
|---|---|---|
| PPG | Photoplethysmography signal | 0 |
| Accel_X | X-axis acceleration | 3 |
| Accel_Y | Y-axis acceleration | 4 |
| Accel_Z | Z-axis acceleration | 7 |
| Temperature | Temperature in °C | 18 |
| GSR Resistance | Skin resistance in kΩ | 20 |
| Gyro_Z | Z-axis angular velocity | 21 |
| Timestamp | Calibrated timestamp in ms | 25 |
| Gyro_X | X-axis angular velocity | 27 |
| Gyro_Y | Y-axis angular velocity | 30 |
| GSR Conductance | Skin conductance in µS | 31 |

The dataset also includes magnetometer, pressure, battery, and additional sensor channels.

### 4. Data Cleaning Pipeline

A preliminary preprocessing pipeline has been outlined to:

- Select calibrated sensor measurements using their original column positions.
- Assign descriptive column names.
- Convert measurement columns to numeric types.
- Identify missing values and inspect summary statistics.
- Convert timestamps from milliseconds to elapsed seconds.
- Inspect physiological signals for artifacts, unusual values, and inconsistencies.

### 5. Exploratory Data Analysis

The planned exploratory analysis includes:

- Plotting PPG, GSR, and temperature measurements over time.
- Examining missing values and signal distributions.
- Comparing recording durations across participants.
- Investigating motion artifacts and sensor variability.

These analyses will help assess signal quality and identify potential preprocessing requirements.

### 6. Preliminary Feature Engineering

We have outlined an initial feature extraction approach using fixed 60-second windows.

**Motion magnitude:**

Three-axis acceleration can be combined into a single magnitude feature:

\[
a = \sqrt{a_x^2 + a_y^2 + a_z^2}
\]

**Window-level features:**

For each 60-second window, the proposed features include:

- Mean and standard deviation of PPG
- Mean and standard deviation of GSR conductance
- Mean and standard deviation of temperature
- Mean and standard deviation of acceleration magnitude

This approach reduces high-frequency sensor recordings into a smaller, structured feature dataset suitable for machine learning.

## Planned Modeling Approach

### General Population Model

Train a baseline classifier using physiological features from multiple participants.

### Personalized Model

Incorporate baseline-relative features to account for differences in physiological measurements across individuals.

### Model Evaluation

Use leave-one-subject-out cross-validation to evaluate generalization to unseen participants.

Planned metrics include:
- ROC-AUC
- F1-score
- Precision and recall
- Specificity and false-positive rate
- Confusion matrix

## Next Steps

1. Complete and validate the preprocessing pipeline on individual recordings.
2. Investigate fasting metadata and determine whether reliable dehydration labels are available.
3. Generate exploratory visualizations and assess signal quality.
4. Extract window-level features across participants.
5. Train an initial population-level baseline model, provided suitable labels are available.
6. Implement personalized baseline-relative features.
7. Compare model performance and investigate false-positive predictions.
8. Analyze feature importance and summarize findings and limitations.

## Tools and Libraries

- Python
- pandas
- NumPy
- Matplotlib
- pathlib
- scikit-learn (planned for modeling)

## Current Status

**Dataset acquisition and initial structural inspection completed.**

The raw CSV format, sensor metadata, calibration indicators, and relevant physiological channels have been identified. Data cleaning, exploratory visualizations, feature extraction, and model training are the next implementation steps.

No validated model results have been reported yet.
