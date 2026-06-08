#!/usr/bin/env python3
"""Main file for testing the Dataset tf_encode method."""

Dataset = __import__('2-dataset').Dataset

data = Dataset()
for pt, en in data.data_train.take(1):
    print(pt, en)
for pt, en in data.data_valid.take(1):
    print(pt, en)
