# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The idea of this project is to build a model that predicts and identify credit card of customers that are most likely to churn.The project was built using python and following PEP8 coding standards while ensuring that the engineering best practices of Machine Learning pipelines are enforced. This project is part of machine learning devOps nanodegree.


## Project Files Structure
* data
    * bank_data.csv
* images
    * eda
        * churn_distribution.png
        * customer_age_distribution.png
        * heatmap.png
        * marital_status_distribution.png
        * total_transaction_distribution.png
    * results
        * feature_importance.png
        * logistics_results.png
        * rf_results.png
        * roc_curve_result.png
* logs
    * churn_library.log
* models
    * logistic_model.pkl
    * rfc_model.pkl
* __init__.py
* .gitignore
* churn_library.py
* churn_notebook.ipynb
* churn_script_logging_and_tests.py
* README.md
* requirements.txt
## Running Files

Clone the project
```
git clone https://github.com/zyadalatar/Predict-Customer-Churn-with-Clean-Code.git
```
Create virtual environment
```
python -m venv venv
```
Actiavte the environment (windows)
```
venv/Scripts/activate
```

Actiavte the environment (Linux)
```
source venv/bin/activate
```
Install the dependencies 
```
pip install -r requirements.txt
```

To test the Project
```
python churn_library.py
python churn_script_logging_and_tests.py
```
Check the best practices are followed:
```
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```
