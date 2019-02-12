"""Utilities for creating nicely-formatted representations of objects."""

import collections
from itertools import islice


MAX_ITERABLE_REPR_LEN = 3
MAX_STRING_REPR_LEN = 50


def obj_to_repr(obj, regular=None, minimized=None):
    """Creates the representation of an object."""
    att_reprs = []

    for att in regular or []:
        att_reprs.append(_attribute_to_repr(att, getattr(obj, att)))

    for att in minimized or []:
        att_reprs.append(
            _attribute_to_repr_minimized(att, getattr(obj, att)))

    return '{}({})'.format(obj.__class__.__name__, ', '.join(att_reprs))


def _obj_to_repr_minimized(obj):
    """Creates the minimized representation of an object."""
    return '{}:{}'.format(obj.__class__.__name__, id(obj))


def _attribute_to_repr(name, value):
    """Creates the representation of an attribute."""
    if value is None:
        return '{}=None'.format(name)
    if isinstance(value, type):
        return '{}={}'.format(name, value.__name__)
    if isinstance(value, str):
        return '{}={}'.format(name, _string_to_repr(value))
    return '{}={!r}'.format(name, value)


def _attribute_to_repr_minimized(name, value):
    """Creates the minimized representation of an attribute."""
    if value is None:
        return '{}=None'.format(name)
    if isinstance(value, collections.Iterable):
        return '{}={}'.format(name, _iterable_to_repr_minimized(value))
    return '{}={}'.format(name, _obj_to_repr_minimized(value))


def _iterable_to_repr_minimized(col):
    """Creates the minimized representation of an iterable."""
    item_reprs = [_obj_to_repr_minimized(item) \
                  for item in islice(col, MAX_ITERABLE_REPR_LEN)]

    if len(col) > MAX_ITERABLE_REPR_LEN:
        item_reprs.append('...')

    return '[{}]'.format(', '.join(item_reprs))


def _string_to_repr(string):
    """Creates the representation of a string."""
    if len(string) > MAX_STRING_REPR_LEN:
        return repr(string[:MAX_STRING_REPR_LEN - 3] + '...')

    return repr(string)
