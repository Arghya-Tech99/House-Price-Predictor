# End-to-End House Price Predictor (MLOps Integrated)

## Project Overview
Most house price prediction projects stop at a Jupyter Notebook. This project takes a "simple" idea and implements it with professional engineering standards. It focuses on building a scalable, readable, and reproducible machine learning pipeline using **ZenML** for orchestration and **MLflow** for experiment tracking.
### Key Features:
- **Core ML:** Rigorous Data Analysis (EDA), Feature Engineering, and Assumption Testing.
- **MLOps:** Automated pipelines for training and deployment.
- **Tracking:** Full experiment tracking and model versioning via MLflow.
- **Design Patterns:** Implementation of Strategy, Template, and Factory patterns for clean code.
- **Continuous Deployment:** Automated model deployment service.

---

## Technical Stack
- **Orchestration:** [ZenML](https://github.com/zenml-io/zenml)
- **Experiment Tracking:** [MLflow](https://mlflow.org/)
- **Data Handling:** Pandas, NumPy, Scikit-learn
- **Framework:** Python 3.13
- **Deployment:** MLflow Model Serving

---

## Pipeline Architecture
The project is structured into modular pipelines:
1. **Ingest Data:** Loading data from source (Kaggle).
2. **Clean Data:** Handling missing values and outliers.
3. **Train Model:** Training the regressor and validating assumptions.
4. **Evaluation:** Comparing performance metrics in MLflow.
5. **Deployment:** Setting up a continuous deployment pipeline that updates the model if performance improves.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Arghya-Tech99/House-Price-Predictor.git](https://github.com/Arghya-Tech99/House-Price-Predictor.git)
   cd House-Price-Predictor

2. **Initialize ZenML:**
   ```bash
   zenml init
   zenml integration install mlflow -y

3. **View the Dashboard:**
   ```bash
   zenml up
   mlflow ui

---

## Future Ideas to Integrate

#### 1. Advanced Feature Engineering
- **Feature Scaling Experiments:** Implement and compare **Standard Scaling** vs. **MinMax Scaling** using MLflow to see which yields better accuracy.
- **Outlier Detection:** Integrate automated outlier detection methods (like Isolation Forests or Z-score filtering) into the data cleaning step.
- **Automated EDA:** Integrate tools like `ydata-profiling` to generate automated data quality reports as part of the ingestion pipeline.

#### 2. Model Robustness & Testing
- **Assumption Testing:** Build automated checks for linear regression assumptions (e.g., Homoscedasticity, Normality of residuals) and have the pipeline trigger a warning if they are violated. 
- **Unit Testing for Data:** Use **Great Expectations** to validate that the incoming data matches the expected schema and quality standards.

#### 3. Deployment & UI
- **Streamlit Frontend:** Build a user-friendly dashboard where users can input house details (rooms, area, etc.) and get an instant price prediction via the MLflow API. 
- **Dockerization:** Wrap the entire training and inference environment in Docker containers for seamless deployment to AWS, GCP, or Azure.
