#!/usr/bin/env python3
"""Test file for the availableShips function."""

availableShips = __import__('0-passengers').availableShips

ships = availableShips(4)
for ship in ships:
    print(ship)
