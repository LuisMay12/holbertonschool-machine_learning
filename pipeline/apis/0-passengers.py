#!/usr/bin/env python3
"""Find Star Wars ships with enough passenger capacity."""

import requests


def availableShips(passengerCount):
    """Return ships that can hold at least ``passengerCount`` passengers."""
    try:
        passenger_count = int(passengerCount)
    except (TypeError, ValueError):
        return []

    ships = []
    url = "https://swapi-api.hbtn.io/api/starships/"

    while url:
        response = requests.get(url)
        data = response.json()

        for ship in data.get("results", []):
            capacity = ship.get("passengers", "")
            try:
                capacity = int(capacity.replace(",", ""))
            except (AttributeError, ValueError):
                continue

            if capacity >= passenger_count:
                ships.append(ship["name"])

        url = data.get("next")

    return ships
