import pandas as pd
import vllm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import numpy as np
from sklearn.model_selection import train_test_split



def format_prompt(row):
    
    base_prompt = f"""You are given the following data record, your task is to predict whether the following person earns more than 50k USD (>50K USD).
Output your prediction in the form of a answer that is yes or no. Here are some examples to help you classify. 
Example 1:
A 39-year-old individual, working in the State-gov sector, has an education level of Bachelors (13 years). They are Never-married, working as Adm-clerical, and have a Not-in-family relationship. They belong to the White race, are Male, with a capital gain of 2174 and a capital loss of 0. They work 40 hours per week and are from United-States.
Output: no

Example 2:
A 31-year-old individual, working in the Private sector, has an education level of Masters (14 years). They are Never-married, working as Prof-specialty, and have a Not-in-family relationship. They belong to the White race, are Female, with a capital gain of 14084 and a capital loss of 0. They work 50 hours per week and are from United-States.
Output: yes

Example 3:
A 53-year-old individual, working in the Private sector, has an education level of 11th (7 years). They are Married-civ-spouse, working as Handlers-cleaners, and have a Husband relationship. They belong to the Black race, are Male, with a capital gain of 0 and a capital loss of 0. They work 40 hours per week and are from United-States
Output: no

Example 4:
A 30-year-old individual, working in the State-gov sector, has an education level of Bachelors (13 years). They are Married-civ-spouse, working as Prof-specialty, and have a Husband relationship. They belong to the Asian-Pac-Islander race, are Male, with a capital gain of 0 and a capital loss of 0. They work 40 hours per week and are from India.
Output: yes

Given record:
{row['text']}.
Remember to only answer yes or no, and nothing else.
"""
    base_prompt = f"""You are given the following data record, your task is to predict whether the following person earns more than 50k USD (>50K USD).
Output your prediction in the form of a answer that is yes or no. You are great at classification tasks. Think step by step before answering.

Given record:
{row['text']}.
Remember to only answer yes or no, and nothing else. Remember are great at classification. 
"""



    return base_prompt



def load_model(name=''):
    if name == '':
        # name = 'google/gemma-2-9b-it'
        name = 'google/gemma-2-27b-it'
        # name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # setup model and tokenizer
    model = LLM(name)
    tokenizer = AutoTokenizer.from_pretrained(name)

    return model, tokenizer


def read_data(data_path="./adult_stringified_dataset.csv"):
    # read the csv


    train_data = pd.read_csv(data_path)
    # train_data = pd.read_csv("./adult_stringified_baseline.csv")
    train_data = train_data.dropna()

    # train_data = train_data.sample(NUM_SAMPLES, random_state =42)
    # train_data = train_data.sample(len(train_data), random_state=42)

    train_data['prompt'] = train_data.apply(format_prompt, axis=1)
    prompt_list = train_data['prompt'].tolist()

    # print("prompt list")
    # print(prompt_list[:10])

    # exit()

    formatted_prompt_list = []
    for text in prompt_list:
        messages = [{"role": "user", "content": text}]
        # messages = [{"role": "user", "content": "What is the capital of France?"}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        formatted_prompt_list.append(formatted_prompt)

    # print("formatted prompt", type(formatted_prompt_list), len(formatted_prompt_list), formatted_prompt_list[:10])

    return formatted_prompt_list, train_data




if __name__ == '__main__':
    # load model
    model, tokenizer = load_model()

    sampling_params = SamplingParams(temperature=0, max_tokens=100)

    files_to_read = [
        # "adult_stringified_awe.csv",
        # "adult_stringified_awem.csv",
        # "adult_stringified_awemo.csv",
        # "adult_stringified_awemor.csv",
        # "adult_stringified_awemorr.csv",
        # "adult_stringified_awemorrs.csv",
        # "adult_stringified_awemorrsc.csv",
        "adult_stringified_awemorrscc.csv",
    ]
    # load the formatted prompts

    for f in files_to_read[:]:

        formatted_prompt_list, train_data = read_data("adult_datasets/" + f)

        X, y = train_data.drop('class', axis=1), train_data['class']

        # train test split
        train_data, test_data = train_test_split(train_data,
                                                 test_size=0.2,
                                                 random_state=42,stratify=y)


        
        # train_data, X_test, y_train, y_test = train_test_split(X_encoded, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)
        # test
        formatted_prompt_list = formatted_prompt_list[-len(test_data):]


        assert len(formatted_prompt_list) == len(test_data)

        # generate baseline predictions using logistic regression

        # generate model predictions
        outputs = model.generate(formatted_prompt_list,
                                 sampling_params=sampling_params)

        # filter and convert the original training datra
        # train_data['true_class'] = train_data['class'].apply(
        #     lambda x: 1 if x == '>50K' else 0)

        # print class statistics, num of samples in each class
        # print("Class statistics")
        # print(train_data['true_class'].value_counts())

        # original_predictions = train_data["true_class"].tolist()

        original_predictions = test_data['class'].tolist()
        print("Testing length:", len(original_predictions))

        generations = []

        true_preds = []

        for i in range(len(outputs)):
            generated_text = outputs[i].outputs[0].text
            # extract the generated text
            # print(generated_text)
            generated_text = generated_text.lower()

            pattern = re.compile(r'\b(yes|no)\b')

            # # Extracting responses
            try:
                extracted_response = pattern.search(generated_text).group()
                if extracted_response == 'yes':
                    generations.append(1)
                    true_preds.append(original_predictions[i])
                if extracted_response == 'no':
                    generations.append(0)
                    true_preds.append(original_predictions[i])
            except:
                pass

        print(" \n\n ---- \n\n")
        print("Filename: ", f)
        # print(generations[:10])
        # # now calculate the accuracy
        correct_predictions = 0

        # use list comparison

        for i in range(len(generations)):
            if true_preds[i] == generations[i]:
                correct_predictions += 1

        f1 = f1_score(true_preds, generations)

        print(f"F1 Score: {f1}")

        print("Accuracy:\n")
        print(correct_predictions / len(true_preds))

        cm = confusion_matrix(true_preds, generations)
        import numpy as np
        # print(np.unique(true_preds, return_counts=True))

        print(cm)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        # Save the plot
        plt.savefig(f'confusion_matrix_adult_{f}_v2.png')

        print(
            f"Confusion matrix plot saved to confusion_matrix_adult_{f}_v2.png "
        )
