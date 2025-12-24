# Device Health Assistant
**B2C Predictive Maintenance System based on Time Series Analysis**

---

## Executive Summary

Device Health Assistant is an end-to-end **AI-powered predictive maintenance system** designed to monitor the health of household electrical devices using time-series data.

The system transforms **raw aggregate energy consumption** into **device-level health insights**, enabling early detection of abnormal behavior and proactive maintenance decisions before critical failures occur.

This project focuses on **real-world constraints**, **interpretable health assessment**, and **product-oriented AI system design**.

---

## Problem Statement

Most household and industrial devices do not fail abruptly.  
Instead, they exhibit **subtle behavioral changes** long before visible breakdowns occur.

Traditional monitoring systems:
- React after failures
- Lack device-level granularity
- Provide binary alerts without context

This project addresses these limitations by:
- Disaggregating aggregate energy data into device-specific signals
- Learning normal device behavior over time
- Quantifying degradation using anomaly scores and health bands
- Presenting results through a user-friendly B2C dashboard

---

## System Architecture

The Device Health Assistant is designed as a modular pipeline that converts raw energy signals into actionable health assessments.

![System Architecture](assets/system_architecture.png)

### Architecture Overview

### 1. Data Source
- Smart meter / main electricity meter
- Provides aggregate household energy consumption as a time series

### 2. Data Preprocessing
- Time indexing and resampling
- Sliding window generation for sequence modeling
- Noise handling and signal smoothing

### 3. Model 1: NILM-Based Device Disaggregation
- Deep learning sequence model (Transformer / sequence-based architecture)
- Input: aggregate power consumption signal
- Output: device-level power consumption
- Purpose: isolate individual device signatures (e.g., refrigerator, HVAC)

### 4. Device-Level Time Series
- Clean and isolated energy signals per device
- Enables device-specific temporal modeling

### 5. Model 2: Time Series Anomaly Detection
- Autoencoder-based anomaly detection model
- Learns normal operational patterns of each device
- Computes reconstruction error as an anomaly signal

### 6. Anomaly Scoring
- Continuous anomaly score over time
- Threshold-based decision logic
- Converts reconstruction error into interpretable severity levels

### 7. Health Assessment
- Binary anomaly flags (Normal / Anomalous)
- Health score bands:
  - **Healthy**
  - **Warning**
  - **Critical**
- Tracks gradual degradation rather than isolated outliers

### 8. User Interface (B2C Dashboard)
- Streamlit-based interactive dashboard
- Time-series visualization per device
- Health indicators and alerts
- Early warning notifications for potential failures

---

## Machine Learning Approach

### NILM (Non-Intrusive Load Monitoring)
- Sequence-based deep learning model
- Learns device-specific power signatures from aggregate signals
- Enables device-level monitoring without additional sensors

### Anomaly Detection
- Autoencoder trained on normal device behavior
- Reconstruction error used as anomaly indicator
- Suitable for scenarios with limited labeled failure data

### Health Scoring
- Continuous anomaly scores aggregated over time
- Health bands provide interpretable, user-friendly feedback
- Focus on trend-based degradation detection

---

## Key Design Decisions

- **Two-stage modeling** (disaggregation + anomaly detection) for modularity
- **Unsupervised learning** to handle scarce failure labels
- **Health bands instead of raw scores** for interpretability
- **Product-oriented design** with a B2C dashboard as a first-class component

---

## Results & Insights

- Successful isolation of device-level energy patterns
- Detection of abnormal behavior before visible failures
- Clear visualization of health degradation trends
- Improved interpretability compared to binary alert systems

*(Example plots and dashboard screenshots can be added in the `assets/` folder.)*

---

## Technologies Used

- Python
- PyTorch
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Streamlit

---

## Future Improvements

- Online / streaming anomaly detection
- Multi-device and multi-household learning
- Adaptive thresholding based on usage patterns
- Deployment as a cloud-based monitoring service
- Integration with notification systems (email / mobile)

---

## License

MIT License
