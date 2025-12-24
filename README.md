# Device Health Monitoring AI

## Overview
An AI-powered device health monitoring system designed to detect early signs of malfunction in electrical appliances using time-series analysis and anomaly detection.

The system focuses on **predictive maintenance**, enabling proactive interventions before critical failures occur.

---

## Problem Statement
Electrical devices often fail silently before visible breakdowns occur. Traditional monitoring approaches rely on reactive alerts rather than predictive insights.

This project aims to:
- Detect abnormal behavior in device power consumption patterns
- Estimate device health status over time
- Provide early warnings for potential failures

---

## System Architecture

The Device Health Assistant is designed as an end-to-end **B2C predictive maintenance system** built on time-series analysis and anomaly detection.

The architecture follows a modular pipeline that transforms raw aggregate energy data into actionable device health insights.

![System Architecture](assets/system_architecture.png)

### Architecture Overview

1. **Data Source**
   - Smart meter / main electricity meter
   - Provides aggregate household energy consumption as a time series

2. **Data Preprocessing**
   - Time indexing and resampling
   - Sliding window generation for sequence modeling
   - Noise handling and signal smoothing

3. **Model 1: NILM-Based Device Disaggregation**
   - Deep learning sequence model (Transformer / sequence-based architecture)
   - Input: aggregate power signal
   - Output: device-level power consumption signals
   - Purpose: isolate individual device signatures (e.g., refrigerator, HVAC)

4. **Device-Level Time Series**
   - Clean, isolated energy signals per device
   - Enables device-specific behavioral modeling

5. **Model 2: Time Series Anomaly Detection**
   - Autoencoder-based anomaly detection model
   - Learns normal temporal behavior of device signals
   - Computes reconstruction error as an anomaly signal

6. **Anomaly Scoring**
   - Continuous anomaly score over time
   - Threshold-based decision logic
   - Converts raw reconstruction error into interpretable scores

7. **Health Assessment**
   - Binary anomaly flags (Normal / Anomalous)
   - Health score bands: Healthy, Warning, Critical
   - Tracks degradation trends instead of single-point failures

8. **User Interface (B2C Dashboard)**
   - Streamlit-based interactive dashboard
   - Time series visualization
   - Health score indicators
   - Alerts and early warning notifications


## Machine Learning Approach
- Time-series modeling
- Unsupervised / semi-supervised anomaly detection
- Health score computation based on anomaly density and severity

---

## Technologies Used
- Python
- PyTorch
- Scikit-learn
- Pandas, NumPy
- Matplotlib / Seaborn
- Streamlit (for visualization)

---

## Results & Insights
- Detection of abnormal device behavior patterns
- Visualization of device health degradation
- Early warning capability before failure events

---

## Future Improvements
- Multi-device learning
- Online / streaming anomaly detection
- Deployment-ready monitoring service

---

## License
MIT License
