#!/usr/bin/env python3
"""Main file for testing semantic_search."""

import os

semantic_search = __import__('3-semantic_search').semantic_search

corpus_path = os.path.join(os.path.dirname(__file__), 'ZendeskArticles')

print(semantic_search(corpus_path, 'When are PLDs?'))
