# Cardiac Disease Prediction & Health Risk Assessment System

An end-to-end Machine Learning web application built using Streamlit, Python, and Scikit-Learn to evaluate cardiovascular disease risk, deliver diagnostic analytics, and provide dynamic model retraining capabilities.

---

## Overview

Cardiovascular diseases are among the leading causes of global mortality. Early detection and risk stratification play a critical role in clinical prevention. This repository provides a modular, production-ready web application designed for interactive risk analysis based on clinical patient metrics.

The system features:
- A user-centric Streamlit web interface with real-time risk assessment.
- Machine learning inference driven by a pre-trained clinical classifier.
- Visual analytics and interactive feature breakdown for patient health metrics.
- Integrated retraining modules that allow users to re-train and update the underlying prediction model dynamically.

---

## Repository Structure

```text
├── assets/                  # Media assets (videos, background imagery, UI banners)
├── Dataset/                 # Raw and processed datasets for model training
├── models/                  # Stored model artifacts and versioning scripts
├── notebooks/               # Jupyter notebooks for exploratory data analysis (EDA)
├── app.py                   # Main Streamlit user interface application
├── backend.py               # Core backend logic, feature engineering, and inference engine
├── cardio_model.pkl         # Trained Machine Learning model artifact
├── requirements.txt         # Project dependencies for python runtime environment
└── README.md                # System documentation
```

---

## Technical Features

### 1. Risk Assessment Engine
- Accepts key physiological parameters such as blood pressure, cholesterol levels, glucose status, smoking history, physical activity, and demographic factors.
- Generates precise probability scores indicating cardiovascular risk severity.

### 2. Interactive Analytics
- Provides graphical representations of risk factors against population benchmarks.
- Highlights primary contributors to elevated risk levels for improved patient education.

### 3. Dynamic Model Retraining
- Allows seamless re-training of the model on newly uploaded datasets directly from the interface.
- Automatically handles pipeline validation and serializes model updates safely.

---

## Installation and Local Setup

### Prerequisites
- Python 3.9 or higher installed on your system.

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mujahidalirattar/Cardiac-Disease-Prediction-Model.git
   cd Cardiac-Disease-Prediction-Model
   ```

2. **Create and Activate Virtual Environment**
   - On Windows:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Application**
   ```bash
   streamlit run app.py
   ```

---

## Deployment on Streamlit Cloud

To deploy this project to Streamlit Community Cloud:

1. Push all project files (excluding virtual environments and cache directories) to your GitHub repository.
2. Sign in to [Streamlit Cloud](https://share.streamlit.io/).
3. Click on **New App** and select the repository: `mujahidalirattar/Cardiac-Disease-Prediction-Model`.
4. Set the main file path to `app.py`.
5. Click **Deploy**.

---

## Author & Contact Information

Developed and maintained by **Mujahid Ali Rattar**.

- **Repository:** [GitHub Repository](https://github.com/mujahidalirattar/Cardiac-Disease-Prediction-Model.git)
- **LinkedIn:** [Mujahid Ali Rattar](https://www.linkedin.com/in/mujahidalirattar/)
- **Email:** bit23f011@gmail.com