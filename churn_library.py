# library doc string
'''
This Module contains function for preprocessing, feature engineering, training, and more.

Author: Zyad

1/30/2023
'''

# import libraries
import logging
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df_data: pandas dataframe
    '''
    try:
        assert isinstance(pth, str)
        df_data = pd.read_csv(pth)
        logging.info('SUCCESS: loading file with path: %s successfully', pth)
        return df_data
    except (FileNotFoundError, AssertionError) as err:
        logging.error('ERROR: loading file with path: %s failed', pth)
        raise err


def preprocess_data(df_data):
    '''
    returns a processed dataframe from the passed dataframe

    input:
            df_data: pandas dataframe
    output:
            df_data: processed pandas dataframe
    '''
    try:
        assert isinstance(df_data, pd.DataFrame)
        df_data['Churn'] = df_data['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        logging.info('SUCCESS: processed the file')
        return df_data
    except (Exception, AssertionError) as err:
        logging.error('ERROR: processing the file failed')
        raise err


def perform_eda(df_data):
    '''
    perform eda on df_data and save figures to images folder
    input:
            df_data: pandas dataframe

    output:
            None
    '''
    try:
        assert isinstance(df_data, pd.DataFrame)
    except (AssertionError) as err:
        logging.error(
            'ERROR: argument is worng type %s', err)
        raise err

    # churning distribution
    plt.figure(figsize=(20, 10))
    df_data['Churn'].hist()
    plt.savefig('images/eda/churn_distribution.png')

    # customer age distribution
    plt.figure(figsize=(20, 10))
    df_data['Customer_Age'].hist()
    plt.savefig('images/eda/age_distribution.png')

    # marital status distribution
    plt.figure(figsize=(20, 10))
    df_data.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/eda/marital_status_distribution.png')

    # TotalTrans_Ct distribution
    plt.figure(figsize=(20, 10))
    sns.distplot(df_data['Total_Trans_Ct'])
    plt.savefig('images/eda/Total_Trans_Ct_distribution.png')

    # correlation between variables in range 0..1
    plt.figure(figsize=(20, 10))
    sns.heatmap(df_data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('images/eda/correlation_between_variables.png')


def encoder_helper(df_data, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df_data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
                       used for naming variables or index y column]

    output:
            df_data: pandas dataframe with new columns for
    '''
    for category in category_lst:
        column_lst = []
        column_groups = df_data.groupby(category).mean()['Churn']

        for val in df_data[category]:
            column_lst.append(column_groups.loc[val])

        if response:
            new_column_name = category + '_' + response
            df_data[new_column_name] = column_lst
        else:
            df_data[category] = column_lst
    return df_data


def perform_feature_engineering(df_data, response):
    '''
    The aim of this function is to perform feature engineering on the data
    input:
              df_data: pandas dataframe
              response: string of response name [optional argument that could be
                         used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    try:
        assert isinstance(df_data, pd.DataFrame)
        assert isinstance(response, str)
    except (AssertionError) as err:
        logging.error('ERROR: argument is wring type %s', err)
        raise err
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df_encoded = encoder_helper(
        df_data=df_data,
        category_lst=cat_columns,
        response=response)
    # target
    target = df_encoded['Churn']
    # Create dataframe
    x_data = pd.DataFrame()
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    x_data[keep_cols] = df_encoded[keep_cols]
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, target, test_size=0.3, random_state=42)
    return (x_train, x_test, y_train, y_test)


def classification_report_image(compute_results_dict):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            compute_results_dict[y_train]: training response values
            compute_results_dict[y_test]:  test response values
            compute_results_dict[y_train_preds_lr]: training predictions from logistic regression
            compute_results_dict[y_train_preds_rf]: training predictions from random forest
            compute_results_dict[y_test_preds_lr]: test predictions from logistic regression
            compute_results_dict[y_test_preds_rf]: test predictions from random forest

    output:
             None
    '''
    # random forest
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                compute_results_dict["y_test"], compute_results_dict["y_test_preds_rf"])),
        {'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                compute_results_dict["y_train"], compute_results_dict["y_train_preds_rf"])),
        {'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/random_results.png')

    # logistic regression
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                compute_results_dict["y_train"], compute_results_dict["y_train_preds_lr"])), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                compute_results_dict["y_test"], compute_results_dict["y_test_preds_lr"])), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/logistic_results.png')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # Save the image
    plt.savefig(output_pth + 'feature_importances.png')


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # plots
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(lrc, x_test, y_test, ax=axis, alpha=0.8)
    plot_roc_curve(cv_rfc.best_estimator_,
                   x_test,
                   y_test,
                   ax=axis,
                   alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.png')

    # Compute and results
    compute_results_dict = {
        "y_train": y_train,
        "y_test": y_test,
        "y_train_preds_lr": y_train_preds_lr,
        "y_train_preds_rf": y_train_preds_rf,
        "y_test_preds_lr": y_test_preds_lr,
        "y_test_preds_rf": y_test_preds_rf}
    classification_report_image(compute_results_dict)
    # Compute and feature importance
    feature_importance_plot(
        model=cv_rfc,
        x_data=x_test,
        output_pth='./images/results/')


if __name__ == '__main__':
    # Import data
    DF = import_data(pth='./data/bank_data.csv')
    # Perform EDA
    perform_eda(DF)
    # Feature engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        DF, response='Churn')

    train_models(
        x_train=X_TRAIN,
        x_test=X_TEST,
        y_train=Y_TRAIN,
        y_test=Y_TEST)
