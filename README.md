
# Machine Learning Projects

## Author
Elgun Ismayilov

## Overview

This repository contains two key machine learning projects focused on addressing real-world problems using advanced predictive models. The projects are:

1. **Loan Default Prediction**  
   Aimed at minimizing financial losses for banks and financial institutions by predicting loan defaults.  
   - **Base Model**: Logistic Regression  
   - **Advanced Models**: SVM, Random Forest, XGBoost, CatBoost, and MLP.  

2. **House Price Prediction**  
   Designed to assist buyers, sellers, and real estate agents in making informed property decisions by predicting house prices.  
   - **Base Model**: Linear Regression  
   - **Advanced Models**: KNN, Decision Tree Regression, LightGBM, and MLP.

---

## Datasets

- **Loan Default Prediction**  
  - **Size**: 87,501 rows and 30 columns  
  - **Features**: Funded amount, location, loan balance, income, credit score, etc.  
  - **Source**: [Kaggle Loan Default Dataset](https://www.kaggle.com/datasets/marcbuji/loan-default-prediction)

- **House Price Prediction**  
  - **Features**: Property size, number of rooms, location, neighborhood, amenities, etc.  
  - **Source**: [Kaggle House Price Dataset](https://www.kaggle.com/datasets/samwash94/dataset-for-house-price-analysis)

---

## Methodology

### Loan Default Prediction
- **Preprocessing**: Label Encoding, Min-Max Scaling  
- **Models Used**: Logistic Regression, SVM, Random Forest, XGBoost, CatBoost, MLP  
- **Evaluation Metrics**: Accuracy, F1-Score, Precision-Recall AUC  

### House Price Prediction
- **Preprocessing**: Label Encoding, Min-Max Scaling  
- **Models Used**: Linear Regression, KNN, Decision Tree Regression, LightGBM, MLP  
- **Evaluation Metrics**: Mean Absolute Error (MAE), Mean Squared Error (MSE), R²  

---

## Results

### Loan Default Prediction
| Model                | Accuracy  | F1 Score | Precision-Recall AUC |
|----------------------|-----------|----------|-----------------------|
| Logistic Regression  | 83.28%    | 0.6184   | 0.6274                |
| SVM                  | 75.38%    | 0.5249   | 0.7538                |
| Random Forest        | 60.74%    | 0.3556   | 0.6074                |
| XGBoost              | 83.95%    | 0.6838   | 0.6242                |
| CatBoost             | 84.03%    | 0.6918   | 0.6243                |

### House Price Prediction
| Model                | MSE       | R²       |
|----------------------|-----------|----------|
| Linear Regression    | 0.001570  | 0.9200   |
| Decision Tree        | 0.000661  | 0.9663   |
| KNN                  | 0.005084  | 0.7409   |
| LightGBM             | 0.000657  | 0.9665   |
| MLP                  | 0.000923  | 0.9530   |

---

## Challenges
- **Data Quality**: Missing values and class imbalance in the loan default dataset.  
- **Model Overfitting**: Observed with Random Forest.  
- **Multicollinearity**: Affected performance in regression models.  
- **Computation Time**: Models like KNN and MLP required significant computational resources.

---

## Recommendations
- **Loan Default Prediction**: XGBoost is recommended for its balance between accuracy and efficiency. CatBoost is a close alternative for categorical data handling.  
- **House Price Prediction**: LightGBM offers the best accuracy and computational efficiency, making it the ideal choice. Decision Tree is also a viable option due to its simplicity and high accuracy.

---

## Conclusion
These projects demonstrate the effectiveness of various machine learning algorithms in solving real-world problems. The findings highlight the importance of model selection, preprocessing techniques, and evaluation metrics in achieving optimal performance.
