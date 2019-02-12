"""Example models for the tests"""

from os.path import join
import pandas as pd
import pypsl as psl


def _read_data_file(path):
    """Loads a TSV file and formats it for pypsl"""
    df = pd.read_csv(path, sep='\t', header=None)
    for i in range(df.shape[1]-1):
        df[i] = df[i].map(str)
    df[df.shape[1]-1] = df[df.shape[1]-1].map(float)

    data = []
    for _, row in df.iterrows():
        data.append([row[col] for col in df])
    return data


def get_friendship_model(data_path):
    # predicates
    similar_pred = psl.Predicate('Similar',
        _read_data_file(join(data_path, 'similar_obs.tsv')))
    block_pred = psl.Predicate('Block',
        _read_data_file(join(data_path, 'location_obs.tsv')))
    friends_pred = psl.Predicate('Friends',
        _read_data_file(join(data_path, 'friends_targets.tsv')), predict=True)

    # prior
    prior = psl.Rule(
        positive_atoms=[],
        negative_atoms=[
            (block_pred, ['P1', 'A']),
            (block_pred, ['P2', 'A']),
            (friends_pred, ['P1', 'P2'])
        ]
    )

    # rules
    rule1 = psl.Rule(
        positive_atoms=[
            (friends_pred, ['P1', 'P2'])
        ],
        negative_atoms=[
            (block_pred, ['P1', 'A']),
            (block_pred, ['P2', 'A']),
            (similar_pred, ['P1', 'P2'])
        ]
    )
    rule2 = psl.Rule(
        positive_atoms=[
            (friends_pred, ['P2', 'P1'])
        ],
        negative_atoms=[
            (block_pred, ['P1', 'A']),
            (block_pred, ['P2', 'A']),
            (friends_pred, ['P1', 'P2'])
        ]
    )
    rule3 = psl.Rule(
        positive_atoms=[
            (friends_pred, ['P1', 'P3'])
        ],
        negative_atoms=[
            (block_pred, ['P1', 'A']),
            (block_pred, ['P2', 'A']),
            (block_pred, ['P3', 'A']),
            (friends_pred, ['P1', 'P2']),
            (friends_pred, ['P2', 'P3'])
        ]
    )

    return psl.Model([
        (10, rule1),
        (10, rule2),
        (1, prior)
    ])


def get_preference_pred_model(data_path):
    # predicates
    avgJokeRating = psl.Predicate('avgJokeRating',
        _read_data_file(join(data_path, 'avgJokeRating.tsv')))
    avgUserRating = psl.Predicate('avgUserRating',
        _read_data_file(join(data_path, 'avgUserRating.tsv')))
    simObsRating = psl.Predicate('simObsRating',
        _read_data_file(join(data_path, 'simObsRating.tsv')))
    isRatingTargets = psl.Predicate('isRatingTargets',
        _read_data_file(join(data_path, 'isRatingTarget.tsv')))
    rating = psl.Predicate('rating',
        _read_data_file(join(data_path, 'ratings.tsv')), predict=True)

    # priors
    prior1 = psl.Rule(
        positive_atoms=[
            (rating, ['U', 'J']),
        ],
        negative_atoms=[
            (isRatingTargets, ['U', 'J'])
        ]
    )
    prior2 = psl.Rule(
        positive_atoms=[],
        negative_atoms=[
            (isRatingTargets, ['U', 'J']),
            (rating, ['U', 'J'])
        ]
    )

    # rules
    rule1 = psl.Rule(
        positive_atoms=[
            (rating, ['U', 'J2']),
        ],
        negative_atoms=[
            (isRatingTargets, ['U', 'J1']),
            (isRatingTargets, ['U', 'J2']),
            (simObsRating, ['J1', 'J2']),
            (rating, ['U', 'J1'])
        ]
    )
    rule2 = psl.Rule(
        positive_atoms=[
            (rating, ['U', 'J']),
        ],
        negative_atoms=[
            (isRatingTargets, ['U', 'J']),
            (avgUserRating, ['U'])
        ]
    )
    rule3 = psl.Rule(
        positive_atoms=[
            (rating, ['U', 'J']),
        ],
        negative_atoms=[
            (isRatingTargets, ['U', 'J']),
            (avgJokeRating, ['J']),
        ]
    )
    rule4 = psl.Rule(
        positive_atoms=[
            (avgUserRating, ['U']),
        ],
        negative_atoms=[
            (isRatingTargets, ['U', 'J']),
            (rating, ['U', 'J'])
        ]
    )
    rule5 = psl.Rule(
        positive_atoms=[
            (avgJokeRating, ['J']),
        ],
        negative_atoms=[
            (isRatingTargets, ['U', 'J']),
            (rating, ['U', 'J'])
        ]
    )

    return psl.Model([
        (1, prior1),
        (1, prior2),
        (1, rule1),
        (1, rule2),
        (1, rule3),
        (1, rule4),
        (1, rule5)
    ])
