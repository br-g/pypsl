"""Utilities for the examples"""

import os
from random import shuffle
import zipfile
import wget
import pandas as pd


# Data paths
REMOTE_PATH = 'https://s3-us-west-1.amazonaws.com/pypsl/examples/{}.zip'
LOCAL_DIR = 'data'


def fetch_data(example_key):
    """Downloads example's data."""
    if not os.path.isdir(LOCAL_DIR):
        os.mkdir(LOCAL_DIR)

    # Download
    local_file = os.path.join(LOCAL_DIR, '{}.zip'.format(example_key))
    if os.path.isfile(local_file):
        os.remove(local_file)
    wget.download(REMOTE_PATH.format(example_key), out=local_file)

    # Unzip
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall(LOCAL_DIR)
    zip_ref.close()


def read_data(path):
    """Loads a TSV data file and formats its content for PyPSL."""
    df = pd.read_csv(path, sep='\t', header=None)
    for i in range(df.shape[1]-1):
        df[i] = df[i].map(str)
    df[df.shape[1]-1] = df[df.shape[1]-1].map(float)

    data = []
    for _, row in df.iterrows():
        data.append([row[col] for col in df])
    return data


def print_data(path, max_rows=3):
    """Prints an extract of some input PyPSL data"""
    data = read_data(path)

    reprs = [str(e) for e in data[:max_rows]]
    if len(data) > max_rows:
        reprs.append('...')

    print('[\n  {}\n]'.format(',\n  '.join(reprs)))


def print_pred(pred, max_rows=3):
    """Prints an extract of some PyPSL predictions."""
    for key in pred:
        val_all = list(pred[key])
        shuffle(val_all)
        val_min = [str(p) for p in val_all[:max_rows]]

        if len(val_all) > max_rows:
            val_min.append('...')

        print("'{}': (\n  {}\n)".format(
            key,
            ',\n  '.join(val_min)
        ))
