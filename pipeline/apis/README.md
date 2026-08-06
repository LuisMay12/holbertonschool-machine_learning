# Data Collection - APIs

This project introduces collecting and transforming data from external APIs
with Python's `requests` package.

## Task 0: Can I join?

`0-passengers.py` defines `availableShips(passengerCount)`. It requests all
pages of the Star Wars API's `starships` resource and returns the names of
ships whose passenger capacity is at least the requested number.

Ships with an unknown or non-numeric passenger capacity are skipped. If no
ship matches, the function returns an empty list.

Example:

```python
availableShips(4)
```

The API's `next` field is used to continue through paginated results until
there are no more pages.
