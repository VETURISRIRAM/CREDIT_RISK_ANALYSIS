"""
@author: Sriram Veturi
@title: Credit Risk Analysis (Predictions of Loan Defaulters)
@date: 04/07/2019
@data: https://www.kaggle.com/wendykan/lending-club-loan-data
"""

# Imports..
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

MONTHS_MAP = {
                'Jan': 1,
                'Feb': 2,
                'Mar': 3,
                'Apr': 4,
                'May': 5,
                'Jun': 6,
                'Jul': 7,
                'Aug': 8,
                'Sep': 9,
                'Oct': 10,
                'Nov': 11,
                'Dec': 12
             }

def get_data(data_file_path):
    """
    This fucntion should just read the data and return the pandas dataframe.
    :param data_file_path: File path where the date is stored.
    :return: df: Dataframe.
    """

    df = pd.read_csv(data_file_path)
    return df


def visualize_class_label(df):
    """
    This function should plot different visualizations of the data.
    :param df: Dataframe
    :return: Mapped Class Label (Just plotting)
    """

    # Class Label Plot
    sns.countplot(df['loan_status'])
    # Class Label Distribution Plot after Mapping
    df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Current': 0, 'Default': 1})
    df["loan_status"].value_counts()
    return df


def preprocess_dataset(df):
    """
    This fucntion should does the initial preprocessing of the dataset
    MAIN PREPROCESSING WOULD BE DONE LATER
    :param df: Dataframe
    :return: df: Preprocessed Dataframe
    """

    # Drop 'ID' column
    df = df.drop(columns=['id'])
    # Remove the string month from term
    df['term'] = df['term'].map({'60 months': 60, '36 months': 36})
    # Update 'issue_m' column
    def update_month(value):

        if type(value) is float:

            return value

        return MONTHS_MAP[value[:3]]

    def update_year(value):

        return int(value[-4:])

    df['issue_m'] = df['issue_d'].apply(lambda month_index: update_month(month_index))
    df['issue_y'] = df['issue_d'].apply(lambda year: update_year(year))
    # Update 'earliest_cr_line' column
    df['earliest_cr_line_m'] = df['earliest_cr_line'].apply(lambda month_index: update_month(month_index))
    def update_cr_line_year(value):

        if type(value) is float:

            return value

        year = int(value[-2:])
        if year > 19:

            year += 1900
        else:

            year += 2000

        return year

    df['earliest_cr_line_y'] = df['earliest_cr_line'].apply(lambda year: update_cr_line_year(year))
    # Update 'emp_length' column
    def update_emp_length(duration):

        if duration == np.nan or duration == 'n/a':

            return np.nan

        if duration == "< 1 year":

            return int(0)

        elif duration == "10+ years":

            return int(10)

        elif type(duration) == str:

            return int(duration[0])

        else:

            return duration

    df['emp_length'] = df['emp_length'].apply(lambda duration: update_emp_length(duration))
    # Update 'last_credit_pull_d' column
    df['last_credit_pull_m'] = df['last_credit_pull_d'].apply(lambda month_index: update_month(month_index))
    df['last_credit_pull_y'] = df['last_credit_pull_d'].apply(lambda year: update_cr_line_year(year))
    # Drop the updated columns
    df = df.drop(columns=['issue_d', 'last_credit_pull_d', 'earliest_cr_line'])
    df['addr_state'].unique()

    # Divide the states by their geographic region
    geography_map = {
                        'west_side': ['CA', 'OR', 'UT', 'WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID'],
                        'south_west': ['AZ', 'TX', 'NM', 'OK'],
                        'south_east': ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN'],
                        'mid_west': ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND'],
                        'north_east': ['CT', 'NY', 'PA', 'NJ', 'RI', 'MA', 'MD', 'VT', 'NH', 'ME']
                    }
    df['geographic_part'] = np.nan
    def map_state(state):

        """
        This function should map the state with the geographic part.
        :param state: US State
        :return: geographic_area
        """

        for geographic_part, states in geography_map.items():

            if state in states:

                return geographic_part


    df['geographic_part'] = df['addr_state'].apply(lambda state: map_state(state))


    return df


def point_plot(df, feature):
    """
    This function should plot the point plot
    :param df: Dataframe
    :param labels: The labels of the parts
    :param feature: Column to plot the point plot
    :return: None (Just plotting)
    """

    plt.figure(figsize=(15, 7))
    g = sns.pointplot(
                      x=feature,
                      y='loan_amnt',
                      data=df,
                      hue='loan_status',
                      join=True,
                      colors=["g", "r"],
                      markers=['o', 'x'],
                      linestyles=['--', '-']
                      )
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.set_xlabel("{}".format(feature.upper()), fontsize=15)
    g.set_ylabel("Loan Amount", fontsize=15)
    g.legend(loc='best')
    g.set_title("Distribution of {} by Loan Defaulters".format(feature.upper()), fontsize=20)
    plt.show()


def dist_plot(df, feature):
    """
    This function shoul plot the distplot.
    :param df: Dataframe
    :param feature: Column to be plotted
    :return: None (Just Plotting)
    """

    feature_values = df[feature].values
    plt.figure(figsize=(10, 5))
    sns.distplot(feature_values, color='red')
    plt.title("{} Distribution".format(feature.upper()))
    plt.xlabel("{}".format(feature.upper()))
    plt.ylabel("Number")


def count_plot(df, feature):
    """
    This function should plot the count plot.
    :param df: Dataframe
    :param feature: Column to be plotted on.
    :return: None (Just Plotting)
    """

    plt.figure(figsize=(15, 5))
    plt.tight_layout()
    sns.countplot(df[feature])
    plt.title('{} Distribution Count Plot'.format(feature.upper()))


def visualize_features(df):
    """
    This function should plot visualizations of different features.
    :param df: Dataframe
    :return:
    """

    # Pie Chart for the Safe and Risky Customers
    labels = "Safe Customers", "Risky Customers"
    plt.figure(figsize=(7, 7))
    df["loan_status"].value_counts().plot.pie(explode=[0, 0.25],
                                              autopct='%1.2f%%',
                                              shadow=True,
                                              colors=["green", "red"],
                                              labels=labels,
                                              fontsize=12,
                                              startangle=70)
    plt.title('Loan Defaulters Distribution', fontsize=15)
    df["emp_length"] = df["emp_length"].fillna(df["emp_length"].mean())
    dist_plot_features = ['loan_amnt', 'installment', 'emp_length']
    for target_feature in dist_plot_features:

        dist_plot(df, target_feature)

    point_plot_features = ['issue_m', 'purpose', 'geographic_part', 'addr_state', 'term']
    for target_feature in point_plot_features:

        point_plot(df, target_feature)

    count_plot_features = ['issue_y', 'verification_status']
    for target_feature in count_plot_features:

        count_plot(df, target_feature)

    fig = plt.figure(figsize=(20, 15))
    df[df['loan_status'] == 1].groupby('addr_state')['loan_status'].count().sort_values().plot(kind='barh')
    plt.ylabel('State', fontsize=15)
    plt.xlabel('Number of loans', fontsize=15)
    plt.title('Number of defaulted loans per state', fontsize=20);

    return df


def feature_selection(df):
    """
    This function finds the most correlating features with the class label
    :param df: DataFrame
    :return: None (Just Printing)
    """

    corr = df.corr()['loan_status'].sort_values()
    print("\n\nMost Important Features:")
    print(corr.tail(5))
    print(corr.head(5))


def fill_missing_values(df):
    """
    This function fills in missing values with the column modes.
    :param df: DataFrame
    :return: df: Filled DataFrame
    """
    missing_values_columns = [
                                'mths_since_last_delinq',
                                'mths_since_last_record',
                                'inq_last_6mths',
                                'inq_last_12m',
                                'last_credit_pull_m',
                                'last_credit_pull_y'
                             ]
    for col in missing_values_columns:

        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def handle_categorical_data(df):
    """
    This function encodes the categorcial features of the dataset.
    :param df: DataFrame
    :return: df: Encoded DataFrame
    """

    # Preprocess categorical columns
    catData = df.select_dtypes(include=['object'])
    catColumns = catData.columns
    df = df.drop(columns=catColumns)
    for x in catData.columns:

        uniqueValues = catData[x].unique()
        mapping = dict(zip(uniqueValues, np.arange(float(len(uniqueValues)))))
        catData[x] = catData[x].map(mapping)

    df = pd.concat([df, catData], axis=1)
    return df


def standardize_dataset(df):
    """
    This function should standardize/normalize the numerical values.
    :param df: DataFrame
    :return: df: Standardized DataFrame
    """

    class_variable = df['loan_status']
    df = df.drop(columns=['loan_status'])
    df_columns = df.columns
    scaler = StandardScaler()
    scaledData = scaler.fit_transform(df)
    df = pd.DataFrame(data=scaledData, columns=df_columns)
    df['loan_status'] = class_variable
    return df


def separate_features_and_labels(df):
    """
    This fucntion should separate features and class labels.
    :param df: DataFrame
    :return: Separated Data
    """

    y = df['loan_status']
    df = df.drop(columns=['loan_status'])
    return df, y


def remove_multicollinear_features(df):
    """
    This fucntion should drop multicollinear features from the dataset.
    :param df: DataFrame
    :return: df: DataFrame with multicollinear features removed
    """

    print("\n\nChecking for the Multi-Collinear Features.")
    # Source: https://stats.stackexchange.com/questions/155028/how-to-systematically-remove-collinear-variables-in-python
    def calculate_vif_(X, thresh=100):

        cols = X.columns
        variables = np.arange(X.shape[1])
        dropped = True
        while dropped:

            dropped = False
            c = X[cols[variables]].values
            vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]
            maxloc = vif.index(max(vif))
            if max(vif) > thresh:

                print('Dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
                variables = np.delete(variables, maxloc)
                dropped = True

        return X[cols[variables]]

    df = calculate_vif_(df)
    print("Removed Multi-Collinear Features.")
    return df

def actual_preprocessing_dataset(df):

    df = fill_missing_values(df)
    df = handle_categorical_data(df)
    df = standardize_dataset(df)
    # df, y = separate_features_and_labels(df)
    df = remove_multicollinear_features(df)
    class_label = df['loan_status']
    df = df.drop(columns=['loan_status'])
    df['loan_status'] = class_label
    return df

def split_dataset(dataset):
    """
    This fucntion should split the dataset in to train and test sets
    :param dataset: dataframe
    :return: X_train, X_test, y_train, y_test (Split sets)
    """

    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size = 0.1)
    return X_train, X_test, y_train, y_test


def balance_dataset(x_train, y_train):
    """
    This function should balance the dataset
    :param x_train: Train Features
    :param y_train: Train Labels
    :return:
    """
    sm = SMOTE(random_state=12, ratio=1.0)
    x_train_res, y_train_res = sm.fit_sample(x_train, y_train)
    return x_train_res, y_train_res


def random_forest_modelling(x_train_res, y_train_res, X_test, y_test):
    """
    This function should provide analysis w.r.t Random Forest Model.
    :param x_train_res: Balanced Train Features
    :param y_train_res: Balanced Train Labels
    :param X_test: Test Features
    :param y_test: Test Labels
    :return: None (Just Printing Stuff)
    """
    print("Random Forest Evaluations")
    print("Cross Validating for best parameters..")
    print("This might take some time..\n")
    clf_rf = RandomForestClassifier()
    estimatorsList = [25, 50]
    parameters = {'n_estimators': estimatorsList}
    gridSearch = GridSearchCV(estimator=clf_rf,
                              param_grid=parameters,
                              scoring="recall",
                              cv=10,
                              n_jobs=4
                              )
    gridSearch.fit(x_train_res, y_train_res)
    bestAccuracyLogBestK = gridSearch.best_score_
    bestParametersLogBestK = gridSearch.best_params_
    print("The best parameters for Random Forest model are :\n{}\n".format(bestParametersLogBestK))
    clf_rf = RandomForestClassifier(n_estimators=50, random_state=12)
    clf_rf.fit(x_train_res, y_train_res)
    print('\nTrain Results')
    print(clf_rf.score(x_train_res, y_train_res))
    print(recall_score(y_train_res, clf_rf.predict(x_train_res)))
    print('\nTest Results')
    print(clf_rf.score(X_test, y_test))
    print(recall_score(y_test, clf_rf.predict(X_test)))


def logistic_regression_modelling(x_train_res, y_train_res, X_test, y_test):
    """
    This function should provide analysis w.r.t Logistic Regression Model.
    :param x_train_res: Balanced Train Features
    :param y_train_res: Balanced Train Labels
    :param X_test: Test Features
    :param y_test: Test Labels
    :return: None (Just Printing Stuff)
    """

    print("\n\n\nLogistic Regression")
    print("Cross Validating for best parameters..")
    print("This might take some time..\n")
    lr = LogisticRegression(multi_class='ovr')
    cList = [1, 10]
    parameters = {'C': cList}
    gridSearch = GridSearchCV(estimator=lr,
                              param_grid=parameters,
                              scoring="recall",
                              cv=10,
                              n_jobs=4)
    gridSearch.fit(x_train_res, y_train_res)
    bestAccuracyLogBestK = gridSearch.best_score_
    bestParametersLogBestK = gridSearch.best_params_
    print("The best parameters for Logistic Regression model are :\n{}\n".format(bestParametersLogBestK))
    lr = LogisticRegression(C=10)
    lr.fit(x_train_res, y_train_res)
    print('\nTrain Results')
    print(lr.score(x_train_res, y_train_res))
    print(recall_score(y_train_res, lr.predict(x_train_res)))
    print('\nTest Results')
    print(lr.score(X_test, y_test))
    print(recall_score(y_test, lr.predict(X_test)))

# Main function starts here..
if __name__ == "__main__":

    df = get_data('application/data.csv')
    df = visualize_class_label(df)
    df = preprocess_dataset(df)
    df = visualize_features(df)
    feature_selection(df)
    df = actual_preprocessing_dataset(df)
    X_train, X_test, y_train, y_test = split_dataset(df)
    x_train_res, y_train_res = balance_dataset(X_train, y_train)
    random_forest_modelling(x_train_res, y_train_res, X_test, y_test)
    logistic_regression_modelling(x_train_res, y_train_res, X_test, y_test)
