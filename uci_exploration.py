from ucimlrepo import fetch_ucirepo
from datasets import Dataset
import pandas as pd
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler

DATA_NAME = 'adult_dataset'
# fetch dataset

if DATA_NAME == 'adult_dataset':

    def perform_logistic_regression(df, scaling=True):
        print(df.columns)
        print(df.shape)


        balance = True
        # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)

        # train_df['class'] = train_df['class'].apply(lambda x: 1 if x == '>50K' else 0)
        # test_df['class'] = test_df['class'].apply(lambda x: 1 if x == '>50K' else 0)


        print("Original dataset columns:")
        print(df.columns)
        
        # Convert class to binary
        df['class'] = df['class'].apply(lambda x: 1 if x == '>50K' else 0)
        
        print("\nOriginal data class statistics:")
        print(df['class'].value_counts())
        
        X = df.drop('class', axis=1)
        y = df['class']

        print(y.head())
        
        if balance:
            # Balance the entire dataset
            rus = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = rus.fit_resample(X, y)
            
            print("\nBalanced data class statistics:")
            print(pd.Series(y_balanced).value_counts())
        else:
            X_balanced, y_balanced = X, y
        
        # concat into a df
        balanced_df = pd.concat([X_balanced, pd.Series(y_balanced, name='class')], axis=1)

        
        # Perform one-hot encoding
        X_encoded = pd.get_dummies(X_balanced, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

        print("\nTraining data class statistics:")
        print(pd.Series(y_train).value_counts())
        
        print("\nTest data class statistics:")
        print(pd.Series(y_test).value_counts())
        

        if scaling is True:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"F1 Score: {f1}")

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return balanced_df
    # To use the function:
    def generate_partial_prompt(row):
        return (
            f"A {row['age']}-year-old individual, working in the {row['workclass']} sector, "
            f"has an education level of {row['education']}. They are {row['marital-status']}, "
            f"working as {row['occupation']}, and have a {row['relationship']} relationship. "
        )

    def gen_partial_prompt_based_on_attributes(row, features):
        # features are strings
        if features == ['age', 'workclass']:
            return (
                f"A {row['age']}-year-old individual, working in the {row['workclass']} sector"
            )
        if features == ['age', 'workclass', 'education']:
            return (
                f"A {row['age']}-year-old individual, working in the {row['workclass']} sector, "
                f"has an education level of {row['education']}."
            )
        if features == ['age', 'workclass', 'education', 'marital-status']:
            return (
                f"A {row['age']}-year-old individual, working in the {row['workclass']} sector, "
                f"has an education level of {row['education']}. They are {row['marital-status']}."
            )
        if features == [
                'age', 'workclass', 'education', 'marital-status', 'occupation'
        ]:
            return (
                f"A {row['age']}-year-old individual, working in the {row['workclass']} sector, "
                f"has an education level of {row['education']}. They are {row['marital-status']}, "
                f"working as {row['occupation']}.")
        if features == [
                'age', 'workclass', 'education', 'marital-status',
                'occupation', 'relationship'
        ]:
            return (
                f"A {row['age']}-year-old individual, working in the {row['workclass']} sector, "
                f"has an education level of {row['education']}. They are {row['marital-status']}, "
                f"working as {row['occupation']}, and have a {row['relationship']} relationship."
            )

        if features == [
                'age', 'workclass', 'education', 'marital-status',
                'occupation', 'relationship', 'race'
        ]:
            return (
                f"A {row['age']}-year-old individual, working in the {row['workclass']} sector, "
                f"has an education level of {row['education']}. They are {row['marital-status']}, "
                f"working as {row['occupation']}, and have a {row['relationship']} relationship."
                f"They belong to the {row['race']} race")
        if features == [
                'age', 'workclass', 'education', 'marital-status',
                'occupation', 'relationship', 'race', 'sex'
        ]:
            return (
                f"A {row['age']}-year-old individual, working in the {row['workclass']} sector, "
                f"has an education level of {row['education']}. They are {row['marital-status']}, "
                f"working as {row['occupation']}, and have a {row['relationship']} relationship."
                f"They belong to the {row['race']} race, are {row['sex']}")

        if features == [
                'age', 'workclass', 'education', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain'
        ]:

            return (
                f"A {row['age']}-year-old individual, working in the {row['workclass']} sector, "
                f"has an education level of {row['education']}. They are {row['marital-status']}, "
                f"working as {row['occupation']}, and have a {row['relationship']} relationship. "
                f"They belong to the {row['race']} race, are {row['sex']}, with a capital gain of {row['capital-gain']}"
            )

        if features == [
                'age', 'workclass', 'education', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                'capital-loss'
        ]:
            return (
                f"A {row['age']}-year-old individual, working in the {row['workclass']} sector, "
                f"has an education level of {row['education']}. They are {row['marital-status']}, "
                f"working as {row['occupation']}, and have a {row['relationship']} relationship. "
                f"They belong to the {row['race']} race, are {row['sex']}, with a capital gain of {row['capital-gain']} and a capital loss of {row['capital-loss']}. "
            )

    def generate_prompt_baseline(row):
        return (
            f"A {row['age']}-year-old individual, working in the {row['workclass']} sector"
        )

    def generate_sentence(row):
        return (
            f"A {row['age']}-year-old individual, working in the {row['workclass']} sector, "
            f"has an education level of {row['education']}. They are {row['marital-status']}, "
            f"working as {row['occupation']}, and have a {row['relationship']} relationship. "
            f"They belong to the {row['race']} race, are {row['sex']}, with a capital gain of {row['capital-gain']} and a capital loss of {row['capital-loss']}. "
            f"They work {row['hours-per-week']} hours per week and are from {row['native-country']}."
        )

    adult = fetch_ucirepo(id=2)

    # data (as pandas dataframes)
    X = adult.data.features
    y = adult.data.targets
    # metadata
    print(adult.metadata)
    # variable information
    print(adult.variables)
    merged_df = pd.concat([X, y], axis=1)

    possible_attribute_combos = [
        ['age', 'workclass'],
        ['age', 'workclass', 'education'],
        ['age', 'workclass', 'education', 'marital-status'],
        ['age', 'workclass', 'education', 'marital-status', 'occupation'],
        [
            'age', 'workclass', 'education', 'marital-status', 'occupation',
            'relationship'
        ],
        [
            'age', 'workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race'
        ],
        [
            'age', 'workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex'
        ],
        [
            'age', 'workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain'
        ],
        [
            'age', 'workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss'
        ],
    ]


    print("Calculating baseline performance using a logistic regression classifier \n")
    # extract few classes based on a possible attribute combination
    logistical_df = merged_df[possible_attribute_combos[-1]]
    logistical_df['class'] = merged_df['income'].copy()

    # perform a baseline classification + get balanced data
    merged_df = perform_logistic_regression(logistical_df) # getting balanced data 
    print('new balanced data', merged_df.head())
    print('new cols', merged_df.columns)


    for features in tqdm(possible_attribute_combos):
        merged_df['text'] = merged_df.apply(
            lambda row: gen_partial_prompt_based_on_attributes(row, features),
            axis=1)

        # merged_df['class'] = merged_df['income'].copy()

        attribute_str = "_".join(features)

        # join the first letter from each word in the attribute string
        attribute_str = "".join([word[0] for word in attribute_str.split("_")])
        attribute_str = attribute_str[:90]

        attribute_str = attribute_str.replace("-", "__")
        #extract the for

        hf_dataset = merged_df[['text', 'class']]
        hf_dataset = Dataset.from_pandas(hf_dataset)
        hf_dataset.push_to_hub(f"adult_dataset_{attribute_str}",
                               token='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')

        hf_dataset.to_csv(
            f"./adult_datasets/adult_stringified_{attribute_str}.csv")



    # merged_df['text'] = merged_df.apply(generate_prompt_baseline, axis=1)
    # merged_df['class'] = merged_df['income'].copy()

    # hf_dataset = merged_df[['text', 'class']]

    # hf_dataset.to_csv("./adult_stringified_baseline.csv")

    # hf_dataset = Dataset.from_pandas(hf_dataset)
    # hf_dataset.push_to_hub("adult_dataset_baseline",
    #                        token='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')
    # print(merged_df.columns)









if DATA_NAME == 'german_credit':
    statlog_german_credit_data = fetch_ucirepo(id=144)

    # # data (as pandas dataframes)
    X = statlog_german_credit_data.data.features
    y = statlog_german_credit_data.data.targets

    # print(X.shape, y.shape)
    # print(X.head())

    # print(statlog_german_credit_data.shape)

    # # # metadata
    # print(statlog_german_credit_data.metadata)

    # # # variable information
    # print(statlog_german_credit_data.variables.columns)

    merged_df = pd.concat([X, y], axis=1)
    print(merged_df.columns)
    # Rename the first five columns
    new_column_names = {
        merged_df.columns[0]: 'CheckingStatus',
        merged_df.columns[1]: 'Duration',
        merged_df.columns[2]: 'CreditHistory',
        merged_df.columns[3]: 'Purpose',
        merged_df.columns[4]: 'CreditAmount',
        merged_df.columns[12]: 'Age',
        merged_df.columns[15]: 'ExistingCredits',
    }

    # prompt = f"""Description: a {age}-year-old individual who has {existing_credits} existing credit(s), a credit amount of {credit_amount}, and a loan duration of {duration} months."""
    if PARTIAL is False:
        truncated_df['text'] = truncated_df.apply(generate_prompt, axis=1)
        hf_dataset = truncated_df[['text', 'class']]

        hf_dataset = Dataset.from_pandas(hf_dataset)
        hf_dataset.push_to_hub("statlog_german_clean",
                               token='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')

    if PARTIAL is True:
        truncated_df['text'] = truncated_df.apply(generate_partial_prompt,
                                                  axis=1)
        print(truncated_df.head())
        hf_dataset = truncated_df[['text', 'class']]
        # print(hf_dataset.head(10))

        hf_dataset = Dataset.from_pandas(hf_dataset)
        hf_dataset.push_to_hub("statlog_german_partial_clean",
                               token='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')

    print(hf_dataset)

    selected_columns = [ 'Duration', 'Age', 'ExistingCredits', 'CreditAmount', 'class']


    PARTIAL = True
    # map values of column class
    merged_df['class'] = merged_df['class'].map({1: 1, 2: 0 })
    merged_df.rename(columns=new_column_names, inplace=True)

    truncated_df = merged_df[selected_columns]


def generate_prompt(row):
    return f"""Description: a {row['Age']}-year-old individual who has {row['ExistingCredits']} existing credit(s), a credit amount of {row['CreditAmount']}, and a loan duration of {row['CreditAmount']} months."""


def generate_partial_prompt(row):
    # return f"""Description: a {row['Age']}-year-old individual who has {row['ExistingCredits']} existing credit(s), a credit amount of {row['CreditAmount']}."""
    return f"""Description: an individual """
