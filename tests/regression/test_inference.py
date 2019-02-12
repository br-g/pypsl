"""End-to-end tests"""

import os
from os.path import join
from collections import defaultdict
import zipfile
import subprocess
import wget
import pytest
import pandas as pd
from models import get_friendship_model, get_preference_pred_model


@pytest.fixture
def psl_reference():
    """Downloads the java implementation of PSL"""
    filename = 'psl-cli-2.1.0.jar'
    path = ('https://linqs-data.soe.ucsc.edu/maven/repositories/psl-releases/'
            'org/linqs/psl-cli/2.1.0/psl-cli-2.1.0.jar')
    if not os.path.isfile(filename):
        wget.download(path, out=filename)


@pytest.fixture
def test_data():
    """Downloads example data"""
    path = ('https://s3-us-west-1.amazonaws.com/pypsl/tests/regression/data.zip')
    if not os.path.isdir('data'):
        wget.download(path, out='data.zip')
        zip_ref = zipfile.ZipFile('data.zip', 'r')
        zip_ref.extractall('.')
        zip_ref.close()


def get_reference_pred(data_dir, filename):
    """Runs the PSL Java reference and returns its output"""
    ret = subprocess.call([
        'java', '-jar', 'psl-cli-2.1.0.jar',
        '--infer',
        '--data', join(data_dir, 'example.data'),
        '--model', join(data_dir, 'example.psl'),
        '--output', join(data_dir, 'pred'),
        '-D', 'admmreasoner.initialconsensusvalue=ATOM',
        '-D', 'admmreasoner.initiallocalvalue=ATOM',
        '-D', 'admmreasoner.objectivebreak=true',
        '-D', 'admmreasoner.epsilonabs=1e-10',
        '-D', 'admmreasoner.epsilonrel=1e-10',
        '-D', 'admmreasoner.maxiterations=3'])
    assert ret == 0

    output = pd.read_csv(join(data_dir, 'pred', filename),
                         sep='\t', header=None)
    for i in range(output.shape[1] - 1):
        output[i] = output[i].apply(lambda x: x.strip("'"))

    pred = defaultdict()
    for _, row in output.iterrows():
        key = tuple(row.iloc[:-1])
        pred[key] = row.iloc[-1]
    return pred


def check_predictions(ref_pred, pred, tolerance):
    """Compare predictions to makes sure they are close to each other"""
    for atom in [val for label_val in pred.values() for val in label_val]:
        key = tuple(atom[:-1])
        value = atom[-1]
        assert abs(ref_pred[key] - value) < tolerance


def test_friendship_example(psl_reference, test_data):
    """Compares the output of pypsl with a reference implementation, on a
       subset of the friendship example"""
    model = get_friendship_model('data/friendship')
    model.ground(check_data=False)
    ref_pred = get_reference_pred('data/friendship', 'FRIENDS.txt')
    pred = model.infer(
        max_iterations=3,
        epsilon_residuals=(1e-10, 1e-10),
        epsilon_objective=1e-10,
        logging_period=1
    )
    check_predictions(ref_pred, pred, 5e-3)


def test_preference_pred_example(psl_reference, test_data):
    """Compares the output of pypsl with a reference implementation, on a
       subset of the preference prediction example"""
    model = get_preference_pred_model('data/preference_prediction')
    model.ground(check_data=False)
    ref_pred = get_reference_pred('data/preference_prediction', 'RATING.txt')
    pred = model.infer(
        max_iterations=3,
        epsilon_residuals=(1e-10, 1e-10),
        epsilon_objective=1e-10,
        logging_period=1
    )
    check_predictions(ref_pred, pred, 1e-5)
