from transformers import pipeline
import pandas as pd
import vllm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from vllm_minimization import read_data

import numpy as np
from sklearn.model_selection import train_test_split


def format_prompt(row):
    base_prompt = f"""You are given the following data record, your task is to predict whether the following person earns more than 50k USD (>50K USD).
    {row['text']}"""
    return base_prompt


def read_data(data_path="./adult_stringified_dataset.csv"):
    # read the csv

    train_data = pd.read_csv(data_path)
    # train_data = pd.read_csv("./adult_stringified_baseline.csv")
    train_data = train_data.dropna()

    train_data['prompt'] = train_data.apply(format_prompt, axis=1)
    prompt_list = train_data['prompt'].tolist()

    return prompt_list, train_data

# Compute the argmax class for each prediction
def compute_argmax_class(predictions):
    results = []
    for pred in predictions:
        max_score_index = pred['scores'].index(max(pred['scores']))
        argmax_class = pred['labels'][max_score_index]
        results.append({
            'sequence': pred['sequence'],
            'predicted_class': 1 if argmax_class == 'income_greater_than_50k' else 0
        })
    return pd.DataFrame(results)


files_to_read = [
    "adult_stringified_awemorrscc.csv",
]
# load the formatted prompts

for f in files_to_read[:]:

    formatted_prompt_list, train_data = read_data("adult_datasets/" + f)

    X, y = train_data.drop('class', axis=1), train_data['class']

    # train test split
    train_data, test_data = train_test_split(train_data,
                                             test_size=0.2,
                                             random_state=42,
                                             stratify=y)

    # train_data, X_test, y_train, y_test = train_test_split(X_encoded, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)
    # test
    formatted_prompt_list = formatted_prompt_list[-len(test_data):]

    original_predictions = test_data['class'].tolist()

    assert len(original_predictions) == len(formatted_prompt_list)

    texts = formatted_prompt_list
    classifier = pipeline(model="sileod/deberta-v3-base-tasksource-nli", device=0)

    results = classifier(
        texts,
        candidate_labels=["income_greater_than_50k", "income_less_than_50k"],
    )

    predictions = compute_argmax_class(results)

    print("Testing length:", len(original_predictions))

    generations = predictions['predicted_class'].tolist()
    true_preds = original_predictions.copy()

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
    plt.savefig(f'confusion_matrix_adult_minilm_{f}_v2.png')

    print(
        f"Confusion matrix plot saved to confusion_matrix_adult_{f}_minilm_v2.png "
    )
