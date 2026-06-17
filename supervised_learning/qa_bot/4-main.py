#!/usr/bin/env python3
"""Main file for testing multi-reference question answering."""

import os

question_answer = __import__('4-qa').question_answer

corpus_path = os.path.join(os.path.dirname(__file__), 'ZendeskArticles')

question_answer(corpus_path)
