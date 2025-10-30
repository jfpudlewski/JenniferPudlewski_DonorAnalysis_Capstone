# Colorado Mesa University Foundation Donor Analytics Project

## Overview
This project explores donor behavior at the **Colorado Mesa University Foundation**, a nonprofit organization supporting student scholarships and institutional development.  
The goal of this analysis is to predict **recurring donor likelihood** using ten years of historical giving data, and to identify the most influential factors that drive donor loyalty and retention.

The analysis was completed entirely in **R** and uses machine learning and statistical modeling techniques, including logistic regression, random forest classification, and donor segmentation through clustering.  
The results support data-informed fundraising strategies and stewardship planning for the Foundation’s advancement team.

---

## Data Description
The dataset used in this analysis consists of **ten years of anonymized donor giving records**.  
Each row represents an individual donor and includes aggregated donation totals, giving streaks, and campaign participation attributes.  

**Key Variables**
- `Recognition.Gift.Total`: Total lifetime giving  
- `Recognition.Gift.Total.This.Year`: Giving amount in the current year  
- `Recognition.Gift.Total.Last.Ten.Years`: Total giving across the last ten fiscal years  
- `Current.Consecutive.Giving.Streak`: Number of consecutive years the donor has contributed  
- `Campaign.ID` and `Constituent.Codes`: Categorical fields describing campaign and donor type  
- `RecurringDonor`: Binary indicator (1 = recurring donor, 0 = one-time donor)

All donor data were anonymized and contain **no personally identifiable information (PII)**.  
The project complies with ethical standards for data protection and nonprofit research.

---

## Methodology
The project follows a structured data science workflow:

1. **Data Cleaning and Exploration**
   - Inspection for missing values, outliers, and multicollinearity  
   - Visualization of donation distributions and giving patterns  

2. **Feature Engineering**
   - Creation of the `RecurringDonor` target variable based on donation streaks  
   - Aggregation of totals and transformations for multi-year features  

3. **Model Development**
   - **Logistic Regression** (base R `glm()`): interpretable classification model predicting recurring donor status.  
   - **Random Forest Classifier** (`randomForest` package): flexible, nonlinear model to compare performance and variable importance.  
   - **K-Means Clustering**: segmentation analysis to identify donor groups by behavior and gift size.

4. **Model Evaluation**
   - Confusion matrices, ROC curves, and AUC metrics to assess predictive power.  
   - Visualization of classification performance and feature importance.

---

## Analysis and Results
- The **logistic regression model** achieved near-perfect discrimination between recurring and one-time donors, with an **AUC close to 1.0**.  
  The most significant predictors were:
  - *Current Consecutive Giving Streak*
  - *Recognition Gift Total Last Ten Years*
  
- The **random forest model** yielded similar accuracy and reinforced the same key predictors, confirming their importance across multiple algorithms.  
  It also uncovered nonlinear relationships, showing that the first few years of giving streak contribute disproportionately to recurrence likelihood.

- A **K-means clustering analysis** revealed three major donor segments:
  1. Long-term high-value recurring donors  
  2. Mid-level consistent annual donors  
  3. One-time or lapsed donors  

These insights help inform targeted outreach campaigns, recognition strategies, and retention initiatives.

---

## Ethical Considerations
All data used in this project were anonymized prior to analysis.  
No personally identifiable information (PII) was accessed or processed.  
The dataset and results are for educational and research purposes only, aligning with the **Association of Fundraising Professionals (AFP) Code of Ethical Standards (2023)**.

---

## File Structure
├── DonorData.csv # Cleaned anonymized donor dataset
├── Donor_Analysis.R # Full R script for analysis
├── logistic_coefficients.csv # Logistic regression output
├── logit_model.rds # Saved logistic regression model
├── rf_model.rds # Saved random forest model
├── roc_logit.png # ROC curve for logistic model
├── rf_varimp.png # Variable importance chart for random forest
├── donor_clusters.png # K-means donor segmentation plot
├── zip_relationship.png # ZIP-level giving regression
└── README.md # Documentation (this file)
## Reproducibility
### Requirements
Install R (version 4.2 or higher) and the following libraries:
```r
install.packages(c("tidyverse", "readxl", "psych", "caret", 
                   "randomForest", "pROC", "ggplot2", "broom"))
source("Donor_Analysis.R")
