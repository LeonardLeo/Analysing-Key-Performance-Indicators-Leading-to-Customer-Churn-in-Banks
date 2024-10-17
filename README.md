# Analysing Key Performance Indicators Leading to Customer Churn in Banks

## Overview

This repository contains the code, data, and reports for analyzing **Key Performance Indicators (KPIs)** that lead to **customer churn** within the banking industry. The project uses **data analytics** and **machine learning** to develop a predictive model aimed at identifying patterns of customer churn, which helps banks improve their retention strategies.

## Key Features

- **Data Preprocessing**: Data cleaning and transformation steps to prepare the dataset.
- **Exploratory Data Analysis (EDA)**: Visualizations and summary statistics to understand churn patterns.
- **Interactive EDA**: Additional notebooks for interactive data exploration.
- **Predictive Model**: A model built using machine learning algorithms such as **Random Forest** and **XGBoost**.
- **Hyperparameter Tuning**: Techniques to optimize the performance of the models.
- **Reporting**: Comprehensive reports in both `.docx` and `.pptx` formats.

## Datasets

- **Customer-Churn-Records.csv**: Contains information about the bank's customers and whether they churned.
- Data is further divided into `generated_data/` for model training.

## Instructions for Use

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/LeonardLeo/Analysing-Key-Performance-Indicators-Leading-to-Customer-Churn-in-Banks.git
   ```

2. Install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Analysis

Use the Jupyter notebooks provided for data exploration and model building:

- Open the notebook: `ANALYSIS OF BANK CHURNERS – Creating a Predictive Model (Notebook).ipynb` for the main analysis.
- For interactive data profiling, check the `pandas_profiling_report/`.

Use the provided Python scripts to automate model building and hyperparameter tuning:

- `model_building.py`: Script to build and evaluate the machine learning models.
- `hyperparameter_tuning.py`: Script to fine-tune model parameters.

## Evaluation

After running the scripts or notebooks, model performance is evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

## Future Work

- Expand the dataset with more diverse customer data.
- Test additional models like deep learning algorithms.
- Implement real-time churn prediction with streaming data.

## Contributing

We welcome any contributions or improvements! Feel free to open an issue or submit a pull request.

