# Asthma Disease Prediction
ML models are effectively used to solve the problem of asthma disease detection with statistical data from a Kaggle dataset. The models were trained and evaluated on a variety of parameters. 
Pre-processing and ensemble techniques such as PCA and bagging were utilized to further increase the performance of the models.

This repository contains a Python script for predicting asthma disease using various machine learning models. The script performs data preprocessing, exploratory data analysis, model training, and evaluation.

## Dataset
The dataset used for this project is stored in `asthma_disease_data.csv`, which includes various patient attributes such as demographics, medical history, environmental exposures, and lung function measurements.

## Models
The script explores a range of machine learning models for asthma prediction, including:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Ada Boost
- XGBoost
- Neural Network (using TensorFlow/Keras)
- Support Vector Machine (SVM)
- Multi-layer Perceptron (MLP)
- Naive Bayes
- K-Nearest Neighbors (KNN)

## Evaluation
The script evaluates the performance of each model using various metrics, including accuracy, precision, recall, F1-score, ROC AUC score, and cross-validation scores. It also includes visualizations such as histograms, correlation heatmaps, countplots, and ROC curves.

:scroll: **Usage**

1. Clone the repository: `git clone https://github.com/your-username/asthma-disease-prediction.git`
2. Install the required libraries: `pip install pandas scikit-learn matplotlib seaborn tensorflow xgboost lightgbm catboost imblearn`
3. Open the Jupyter Notebook `lr_asthma.ipynb` in your Jupyter environment.
4. Run the cells in the notebook.
## Results
The script outputs a comparison table summarizing the performance of each model based on various metrics. It also generates visualizations to aid in understanding the data and model performance.

## Note
The script may require adjustments depending on the specific dataset and desired analysis. The provided code serves as a starting point for asthma disease prediction and can be further customized and extended.
