#!/usr/bin/env python3
"""Print a GitHub user's location using the GitHub API."""

import sys
import time

import requests


def main():
    """Fetch and print the location of the GitHub user in the URL argument."""
    response = requests.get(sys.argv[1])

    if response.status_code == 200:
        print(response.json().get("location"))
    elif response.status_code == 403:
        reset = int(response.headers["X-Ratelimit-Reset"])
        minutes = int((reset - time.time()) / 60)
        print("Reset in {} min".format(minutes))
    else:
        print("Not found")


if __name__ == "__main__":
    main()
