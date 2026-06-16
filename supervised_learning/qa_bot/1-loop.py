#!/usr/bin/env python3
"""Basic question-answer loop."""


def question_loop():
    """Prompt the user for questions until they choose to exit."""
    exit_words = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        question = input('Q: ')

        if question.lower() in exit_words:
            print('A: Goodbye')
            break

        print('A:')


if __name__ == '__main__':
    question_loop()
