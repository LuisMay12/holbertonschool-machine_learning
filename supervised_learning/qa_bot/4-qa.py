#!/usr/bin/env python3
"""Question answering over multiple reference documents."""

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


QA_TOKENIZER = None
QA_MODEL = None
SEARCH_MODEL = None


def load_qa():
    """Load and cache the tokenizer and BERT QA model."""
    global QA_TOKENIZER
    global QA_MODEL

    if QA_TOKENIZER is None:
        QA_TOKENIZER = BertTokenizer.from_pretrained(
            'bert-large-uncased-whole-word-masking-finetuned-squad'
        )

    if QA_MODEL is None:
        QA_MODEL = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    return QA_TOKENIZER, QA_MODEL


def load_search():
    """Load and cache the semantic search model."""
    global SEARCH_MODEL

    if SEARCH_MODEL is None:
        SEARCH_MODEL = hub.load(
            'https://tfhub.dev/google/universal-sentence-encoder-large/5'
        )

    return SEARCH_MODEL


def semantic_search(corpus_path, sentence):
    """Find the document in a corpus most similar to a sentence.

    Args:
        corpus_path (str): path to the corpus of reference documents
        sentence (str): sentence used to perform semantic search

    Returns:
        str: reference text of the most similar document
    """
    documents = []

    for filename in sorted(os.listdir(corpus_path)):
        if filename.endswith('.md'):
            path = os.path.join(corpus_path, filename)
            with open(path, encoding='utf-8') as f:
                documents.append(f.read())

    model = load_search()
    embeddings = model(documents + [sentence])

    doc_embeddings = embeddings[:-1]
    sentence_embedding = embeddings[-1]

    similarities = np.inner(doc_embeddings, sentence_embedding)
    best_match = np.argmax(similarities)

    return documents[best_match]


def answer_question(question, reference):
    """Find a text snippet in reference that answers question.

    Args:
        question (str): question to answer
        reference (str): reference document containing the answer

    Returns:
        str: answer found in the reference, or None if no answer is found
    """
    tokenizer, model = load_qa()

    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    tokens = ['[CLS]'] + question_tokens + ['[SEP]']
    input_type_ids = [0] * len(tokens)

    tokens += reference_tokens + ['[SEP]']
    input_type_ids += [1] * (len(reference_tokens) + 1)

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)

    inputs = [
        tf.expand_dims(tf.constant(input_word_ids), 0),
        tf.expand_dims(tf.constant(input_mask), 0),
        tf.expand_dims(tf.constant(input_type_ids), 0)
    ]

    outputs = model(inputs)
    start = int(tf.argmax(outputs[0][0][1:]) + 1)
    end = int(tf.argmax(outputs[1][0][1:]) + 1)

    if start == 0 or end == 0 or start > end:
        return None

    answer_tokens = tokens[start:end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if answer == '' or answer in ('[CLS]', '[SEP]'):
        return None

    return answer


def question_answer(corpus_path):
    """Answer questions from the best matching reference document."""
    exit_words = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        question = input('Q: ')

        if question.lower() in exit_words:
            print('A: Goodbye')
            break

        reference = semantic_search(corpus_path, question)
        answer = answer_question(question, reference)

        if answer is None:
            print('A: Sorry, I do not understand your question.')
        else:
            print('A: {}'.format(answer))
