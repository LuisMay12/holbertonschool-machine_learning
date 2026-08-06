#!/usr/bin/env python3
"""Find the home planets of sentient species in the Star Wars API."""

import requests


def sentientPlanets():
    """Return the names of the home planets of all sentient species."""
    planets = []
    url = "https://swapi-api.hbtn.io/api/species/"

    while url:
        response = requests.get(url)
        data = response.json()

        for species in data.get("results", []):
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()

            if classification != "sentient" and designation != "sentient":
                continue

            homeworld = species.get("homeworld")
            if not homeworld:
                continue

            planet = requests.get(homeworld).json()
            planets.append(planet["name"])

        url = data.get("next")

    return planets
