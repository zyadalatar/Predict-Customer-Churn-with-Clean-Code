'''
This Module contains the test function for preprocessing, feature engineering, training, and more.

Author: Zyad

2/1/2023
'''
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Success: Testing import_data")
    except FileNotFoundError as err:
        logging.error("ERROR: Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "ERROR: Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_preprocess_data(preprocess_data):
    '''
    test data preprocess - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        df = preprocess_data(df)
        logging.info("Success: Testing preprocess_data")
    except (Exception, AssertionError) as err:
        logging.error("ERROR: Testing preprocess_eda")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    df_data = cls.import_data("./data/bank_data.csv")
    df_data = cls.preprocess_data(df_data=df_data)
    try:
        perform_eda(df_data=df_data)
        logging.info("Success: perform_eda")
    except KeyError as err:
        logging.error('Column "%s" not found', err.args[0])
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    df_data = cls.import_data("./data/bank_data.csv")
    df_data = cls.preprocess_data(df_data=df_data)
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    try:
        encoder_helper(df_data=df_data,
                       category_lst=cat_columns,
                       response=None)
        logging.info("Success encoder_helper")
    except AssertionError as err:
        logging.error("Error Testing encoder_helper")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    df_data = cls.import_data("./data/bank_data.csv")
    df_data = cls.preprocess_data(df_data=df_data)
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df_encoded = cls.encoder_helper(
        df_data=df_data,
        category_lst=cat_columns,
        response=None)
    try:

        perform_feature_engineering(df_encoded, response='Churn')
        logging.info("Success perform_feature_engineering")
    except Exception as err:
        logging.error("ERORR: Testing perform_feature_engineering")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    df_data = cls.import_data("./data/bank_data.csv")
    df_data = cls.preprocess_data(df_data=df_data)
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df_encoded = cls.encoder_helper(
        df_data=df_data,
        category_lst=cat_columns,
        response=None)

    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        df_encoded, response='Churn')

    try:
        train_models(x_train, x_test, y_train, y_test)
        logging.info("Success test_train_models")
    except Exception as err:
        logging.error("ERORR: Testing test_train_models")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_preprocess_data(cls.preprocess_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
