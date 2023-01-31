# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The idea of this project is to build a model that predicts and identify credit card of customers that are most likely to churn.The project was built using python and following PEP8 coding standards while ensuring that the engineering best practices of Machine Learning pipelines are enforced. This project is part of machine learning devOps nanodegree.

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
