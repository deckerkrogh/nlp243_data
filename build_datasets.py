import pandas as pd
from sklearn.model_selection import train_test_split


def build_t1_data():
    url = 'https://raw.githubusercontent.com/deckerkrogh/nlp243_data/main/MaSaC_train_erc.json'
    raw_data = pd.read_json(url)
    train, test = train_test_split(raw_data, test_size=0.2, random_state=1)
    test, dev = train_test_split(test, test_size=0.5, random_state=1)

    train.to_json('datasets/task1_train.json', orient='records')
    test.to_json('datasets/task1_test.json', orient='records')
    dev.to_json('datasets/task1_dev.json', orient='records')


def build_t2_data():
    url = 'https://raw.githubusercontent.com/deckerkrogh/nlp243_data/main/MaSaC_train_efr.json'
    raw_data = pd.read_json(url)
    train, test = train_test_split(raw_data, test_size=0.2, random_state=1)
    test, dev = train_test_split(test, test_size=0.5, random_state=1)

    train.to_json('datasets/task2_train.json', orient='records')
    test.to_json('datasets/task2_test.json', orient='records')
    dev.to_json('datasets/task2_dev.json', orient='records')


def build_t3_data():
    url = 'https://raw.githubusercontent.com/deckerkrogh/nlp243_data/main/MELD_train_efr.json'
    raw_data = pd.read_json(url)

    train, test = train_test_split(raw_data, test_size=0.2, random_state=1)
    test, dev = train_test_split(test, test_size=0.5, random_state=1)

    train.to_json('datasets/task3_train.json', orient='records')
    test.to_json('datasets/task3_test.json', orient='records')
    dev.to_json('datasets/task3_dev.json', orient='records')

    # For ensuring data is well-structured
    #with open('datasets/task3_train.json', 'r') as file:
        #d = pd.read_json(file)
        #print(d['utterances'][0])


build_t1_data()
build_t2_data()
build_t3_data()
