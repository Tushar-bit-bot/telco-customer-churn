# Telco Customer Churn Prediction

A beginner supervised machine learning project using the Kaggle Telco Customer Churn dataset. The project explores customer churn patterns and trains classification models to estimate whether a customer may leave.

## Project questions

- Which customer groups have higher churn rates?
- How are contract type, tenure, monthly charges, and services related to churn?
- How well can a model identify customers who churn?

## Dataset

The dataset contains 7,043 customer records and 21 columns. The target is `Churn` (`Yes` means the customer left; `No` means they stayed).

Dataset: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

`TotalCharges` was stored as text and contained 11 blank entries. Those rows had zero tenure, so the blanks were treated as zero charges before modeling. The project also checked for null values, duplicate records, and duplicate customer IDs.

## Workflow

1. Load and inspect the data with pandas.
2. Clean `TotalCharges` and check data quality.
3. Explore churn rates by contract, tenure group, monthly charges, internet service, tech support, and payment method.
4. Separate predictors (`X`) from the target (`y`) and exclude `customerID`.
5. Split data into training and test sets, preserving the class proportions.
6. Build preprocessing and classification steps in a scikit-learn pipeline. Categorical features are one-hot encoded and numeric features are scaled.
7. Compare Logistic Regression and Random Forest with stratified 5-fold cross-validation.
8. Address class imbalance, tune Logistic Regression with GridSearchCV, and evaluate the selected model on the held-out test set.
9. Interpret model coefficients and summarize possible business actions.

## EDA findings

| Customer group | Churn rate |
|---|---:|
| Month-to-month contract | 42.7% |
| One-year contract | 11.3% |
| Two-year contract | 2.8% |
| Tenure 0–12 months | 47.4% |
| Tenure 13–24 months | 28.7% |
| Tenure 25+ months | 14.0% |
| Fiber optic service | 41.9% |
| DSL service | 19.0% |
| No internet service | 7.4% |
| No tech support | 41.6% |
| Tech support | 15.2% |
| Electronic check | 45.3% |

Customers who churned had an average tenure of about 18 months, compared with about 37.6 months for customers who stayed. Their average monthly charge was about $74.44, compared with $61.27 for customers who stayed.

These are associations in this dataset; they do not establish that any factor caused churn.

## Model results

The tuned, class-weighted Logistic Regression was selected using stratified 5-fold cross-validation and F1 scoring for the churn (`Yes`) class. Grid search selected `C = 0.01` with mean cross-validation F1 of about 0.63.

On the held-out test set (1,409 customers):

| Metric | Churn (`Yes`) result |
|---|---:|
| Precision | 0.51 |
| Recall | 0.79 |
| F1 score | 0.62 |
| Overall accuracy | 0.74 |

The model identified 295 of 374 actual churners and missed 79. It also predicted churn for 282 customers who stayed. This model prioritizes finding more churners, with a corresponding increase in false alarms. A business would need to weigh outreach costs against the value of identifying at-risk customers.

## Business interpretation

The analysis suggests that newer and month-to-month customers may be useful groups for early retention outreach. The company could investigate the service and billing experience of customers with higher observed churn rates, then test low-cost support or check-in actions in a pilot. The findings should be treated as hypotheses to test, not proof of cause.

## Run the project

1. Install Python and Jupyter Notebook.
2. Install the required packages:

   ```bash
   pip install pandas matplotlib seaborn scikit-learn kagglehub notebook
   ```

3. Open `telco_project_organized.ipynb` in Jupyter.
4. Run the cells from top to bottom. KaggleHub downloads the public dataset when the notebook runs.

## Files

- `telco_project_organized.ipynb` — cleaned, sectioned notebook with the full analysis and modeling workflow.
- `README.md` — project overview, methods, and results.

## Limitations and next steps

- Model scores depend on this dataset and may not generalize to current customers or another telecom provider.
- Coefficients show model associations, not causal effects.
- The tuned model's 0.51 churn precision means many predicted at-risk customers did not churn in the test set.
- Future work could compare decision thresholds, add more classifiers, check calibration and fairness, and validate retention actions with a controlled pilot.


