#!/usr/bin/env python3
"""Test file for the sentientPlanets function."""

sentientPlanets = __import__('1-sentience').sentientPlanets

planets = sentientPlanets()
for planet in planets:
    print(planet)
