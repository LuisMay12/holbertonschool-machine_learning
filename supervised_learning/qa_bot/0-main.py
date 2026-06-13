#!/usr/bin/env python3
"""Main file for testing question_answer."""

question_answer = __import__('0-qa').question_answer

with open('ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

print(question_answer('When are PLDs?', reference))
