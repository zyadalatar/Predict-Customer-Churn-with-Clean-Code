# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The idea of this project is to build a model that predicts and identify credit card customers that are most likely to churn.The project was built using python and following  PEP8 coding standards and ensuring the engineering best practices of Machine Learning are enforced as MLOps is an evolving field 

## Running Files

Clone the project
```
git clone 
```
Create virtual environment
```
python -m venv venv
```
Actiavte the environment (windows)
```
venv/Scripts/activate
```
Install the dependencies 
```
pip install -r requirements
```

To test the Project
```
python churn_library.py
python_script_logging_and_tests.py
```
Check the best practices are followed:
```
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```
