import os
import json
import argparse
import ganbert
import pandas as pd
from sklearn.model_selection import train_test_split


output_folder = 'results'


def loda_data(input_file, text_label, label):
    data = []
    label_list = set()
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            content = json.loads(line)
            text = content[text_label]
            target_label = content[label]
            data.append((text, target_label))
            label_list.add(target_label)
    label_list.add('unknow')
    return data, label_list

def load_df(df, text_label, label):
    data = []
    for text, target_label in zip(df[text_label], df[label]):
        data.append((text, target_label))
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default="data/letters/classification/classifier_data_train.json",
                        help='The training data for the classifier.')
    parser.add_argument('--input',
                        default="data/letters/classification/classifier_data_eval.json",
                        help='The text data to use for evaluation (one json per line)')
    parser.add_argument('--output',
                        default=os.path.join(output_folder, "results_GANBERT.txt"),
                        help='Folder where to write the classifier evaluation results')
    parser.add_argument('--text_label',
                        default="text",
                        help='Label/field in the json data which contains the text to classify.'
                             'Can also be multiple labels separated by comma (,)')
    parser.add_argument('--label',
                        default="author",
                        help='Label/field to use for training and classification')
    args = parser.parse_args()

    # here to change the annotated rate
    # unlabeled_data, labeled_data = train_test_split(loda_data(args.training, args.text_label, args.label)[0],
    #                                                 test_size=5e-3, random_state=42)
    # unlabeled_data = [(text, 'unknow') for text, label in unlabeled_data]

    df_train = pd.read_json(args.training, lines=True)
    frames = []
    # here set the annotated rate:
    annotated_rate = 1e-3
    n = round(annotated_rate * df_train.shape[0] / len(df_train[args.label].unique()))

    for author in df_train[args.label].value_counts().index:
        frames.append(df_train.loc[df_train[args.label] == author].sample(n, random_state=42))
    labeled_df = pd.concat(frames)
    labeled_data = load_df(labeled_df, args.text_label, args.label)
    unlabel_df = df_train[~df_train.index.isin(labeled_df.index)]
    unlabeled_data = load_df(unlabel_df, args.text_label, args.label)
    unlabeled_data = [(text, 'unknow') for text, label in unlabeled_data]

    test_data, label_list = loda_data(args.input, args.text_label, args.label)

    model = ganbert.Ganbert(label_list)
    model.train(labeled_data, unlabeled_data, test_data, label_list, args.output, args.label)
