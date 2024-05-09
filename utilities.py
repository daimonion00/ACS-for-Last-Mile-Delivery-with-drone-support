"""
Author: Georg Börries
Date: 24.04.2024

Dieses Modul enthält Hilfsfunktionen
"""
import random
import numpy as np


def generate_delivery_coordinates(depot: tuple[int, int],
                                  num_points: int,
                                  max_distance: int) -> list[tuple[float, float]]:
    """
    Funktion zum Generieren zufälliger Lieferkoordinaten in einem Liefergebiet mit vorgegebener Größe
    :param depot: Position des Depots
    :param num_points: Anzahl der zu erzeugenden Lieferkoordinaten
    :param max_distance: maximale Entfernung vom Ursprung
    :return:
    """
    # Liste für die Koordinaten
    coordinates = [depot]

    while len(coordinates) < num_points + 1:
        # Zufällige x- und y-Koordinate erzeugen
        x = round(random.uniform(-max_distance, max_distance), 1)
        y = round(random.uniform(-max_distance, max_distance), 1)

        # Prüfen, ob das Koordinatenpaar schon vorhanden ist
        if (x, y) not in coordinates:
            # Koordinatenpaar zur Liste hinzufügen
            coordinates.append((x, y))

    return coordinates


def calculate_tour_distance(tour: list[int], distances: np.ndarray) -> float:
    """
    Hilfsfunktion zum Berechnen der Distanz für eine gegebene Tour
    :param tour: Liste mit der Tour
    :param distances: Matrix mit den Distanzen zwischen den Koordinaten
    :return: zurückgelegte Gesamtdistanz
    """
    total_distance = 0

    for i in range(len(tour) - 1):
        start_point = tour[i]
        end_point = tour[i + 1]
        total_distance += distances[start_point, end_point]

    return total_distance


def split_tour(tour: list[int]) -> (list[list[int]], list[str], list[str]):
    """
    Funktion zum Aufteilen einer Tour mit Rückfahrten zum Depot in mehrere Einzeltouren
    :param tour: Liste mit der Tour
    :return: Liste mit den Einzeltouren, Liste mit den Bezeichnungen der Einzeltouren und Liste mit dem Plot-Stil für
                die Touren
    """
    # Listen für die Rückgabe initialisieren
    split_tours = []
    names = []
    styles = []

    # Einzeltouren ermitteln
    index_depot = []  # Liste für die Indizes die einem Aufenthalt beim Depot entsprechen
    # Depotaufenthalte ermittlen
    for i in range(len(tour)):
        if tour[i] == 0:
            index_depot.append(i)
    # Tour anhand der Depot-Indizes in Einzeltouren aufteilen
    for i in range(len(index_depot) - 1):
        split_tours.append(tour[index_depot[i]:index_depot[i + 1] + 1])
        # Name und Plot-Stil für die Einzeltour festlegen
        names.append(f'Tour {i + 1}')
        styles.append('o-')

    return split_tours, names, styles
