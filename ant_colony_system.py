"""
Author: Georg Börries
Datum: 12.05.2024

Klasse für das Ant Colony System zum Optimieren von Liefertouren mit und ohne Drohnenunterstützung

Grundlage für das ACS geklont von https://github.com/Akavall/AntColonyOptimization und überarbeitet für die Verwendung
zur Optimierung von Liefertouren mit Drohnenunterstützung
"""
import random
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist

from utilities import calculate_tour_distance, split_tour


class AntColonySystem(object):

    def __init__(self,
                 coordinates: list[tuple[float, float]] | np.ndarray,
                 n_ants: int,
                 alpha: float,
                 beta: float,
                 rho: float,
                 q0: float,
                 max_iterations: int,
                 max_duration: int,
                 capacity_truck: int,
                 capacity_drone: int,
                 velocity_truck: int,
                 velocity_drone: int,
                 cost_per_km_truck: float,
                 cost_per_km_drone: float,
                 cost_per_h_truck: float,
                 cost_per_h_drone: float,
                 loading_time: float,
                 delivery_time: float) -> None:
        """
        Initialisierung des Ant Colony Systems mit den übergebenen Parametern
        :param coordinates: Liste mit den Koordinaten
        :param n_ants: Anzahl der Ameisen zur Erzeugung von Lösungen
        :param alpha: Gewichtung der Pheromonkonzentration
        :param beta: Gewichtung der heuristischen Informationen
        :param rho: Parameter für den Pheromon-Verfall
        :param q0: Grenzwert für die Auswahlmethode (Exploitation ≤ q0 < Exploration)
        :param max_iterations: maximale Iterationsanzahl
        :param max_duration: maximale Optimierungsdauer
        :param capacity_truck: Kapazität des Lieferwagens (Stück)
        :param capacity_drone: Kapazität der Drohne (Stück)
        :param velocity_truck: Geschwindigkeit des Lieferwagens (km/h)
        :param velocity_drone: Geschwindigkeit der Drohne (km/h)
        :param cost_per_km_truck: Kosten pro km für den Lieferwagen
        :param cost_per_km_drone: Kosten pro km für die Drohne
        :param cost_per_h_truck: Kosten pro h für den Lieferwagen
        :param cost_per_h_drone: Kosten pro h für die Drohne
        :param loading_time: benötigte Zeit zum Beladen des Lieferfahrzeugs/der Drohne pro Paket
        :param delivery_time: benötigte Zeit zum Ausliefern am Lieferort pro Paket
        """
        # Parameter zuweisen
        self.coordinates = coordinates
        self.destinations = len(coordinates) - 1
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.max_iterations = max_iterations
        self.max_duration = max_duration
        self.capacity_truck = capacity_truck
        self.capacity_drone = capacity_drone
        self.cost_per_km_truck = cost_per_km_truck
        self.cost_per_km_drone = cost_per_km_drone
        self.cost_per_h_truck = cost_per_h_truck
        self.cost_per_h_drone = cost_per_h_drone
        self.velocity_truck = velocity_truck
        self.velocity_drone = velocity_drone
        self.loading_time = loading_time
        self.delivery_time = delivery_time
        # Distanzen berechnen
        self.distances_truck = np.array(squareform(pdist(self.coordinates, 'cityblock')))
        self.distances_drone = np.array(squareform(pdist(self.coordinates, 'euclidean')))
        # Distanzen der Koordinaten zu sich selbst auf unendliche setzen, um Division durch null zu verhindern
        np.fill_diagonal(self.distances_truck, np.inf)
        np.fill_diagonal(self.distances_drone, np.inf)
        # heuristische Information der Distanzen über den Kehrwert bilden
        self.visibility_truck = 1 / self.distances_truck
        self.visibility_drone = 1 / self.distances_drone
        # tau_0 initialisieren
        self.tau_0 = 1.0
        self.tau_min = ((self.tau_0 * (1 - 0.05 ** (1 / self.destinations))) /
                        (((self.destinations / 2) - 1) * 0.05 ** (1 / self.destinations)))
        # Pheromon-Matrizen initialisieren
        self.pheromones_truck = np.full(self.distances_truck.shape, self.tau_0)
        self.pheromones_drone = np.copy(self.pheromones_truck)
        # Parameter zum Steuern, ob bei der Generierung der Touren die Nachlieferung/Rückfahrt zum Depot erzwungen
        # werden soll
        self.force_best = False
        # Parameter zum Steuern, ob Pheromonkonzentration nach Min-Max AS eingegrenzt werden soll
        self.use_min_max = False

    def reinit_pheromones(self):
        """
        Funktion zum erneuten Initialisieren der Pheromon-Matrizen
        """
        self.tau_min = ((self.tau_0 * (1 - 0.05 ** (1 / self.destinations))) /
                        (((self.destinations / 2) - 1) * 0.05 ** (1 / self.destinations)))
        self.pheromones_truck = np.full_like(self.pheromones_truck, self.tau_0, dtype = float)
        self.pheromones_drone = np.full_like(self.pheromones_drone, self.tau_0, dtype = float)

    def reset_pheromones(self):
        """
        Funktion zum resetten der Pheromon-Matrizen
        """
        self.tau_0 = 1.0
        self.pheromones_truck.fill(self.tau_0)
        self.pheromones_drone.fill(self.tau_0)

    def pheromone_update_local(self, pheromone_matrice: np.ndarray, current_node: int, next_node: int) -> np.ndarray:
        """
        Funktion zum Berechnen der lokalen Pheromon-Aktualisierung auf der Kante zwischen aktuellem und nächsten Knoten
        :param pheromone_matrice: Pheromon-Matrix
        :param current_node: aktueller Knoten
        :param next_node: nächster Knoten
        :return: aktualisierte Pheromonkonzentration
        """
        new_pheromones = (1 - self.rho) * pheromone_matrice[current_node, next_node] + (self.rho * self.tau_0)
        return new_pheromones

    def pheromone_update_global(self,
                                pheromone_matrice: np.matrix,
                                tour: list[int],
                                cost: float) -> np.matrix:
        """
        Funktion zum Durchführen des globalen Pheromon-Updates
        :param pheromone_matrice: Pheromon-Matrix
        :param tour: global beste Tour
        :param cost: Kosten der global besten Tour
        :return: Pheromon-Matrix mit den aktualisieren Pheromon-Konzentrationen
        """
        pheromone_matrice = (1 - self.rho) * pheromone_matrice
        for i in range(len(tour) - 1):
            current_node = tour[i]
            next_node = tour[i + 1]
            pheromone_matrice[current_node, next_node] += self.rho * (1 / cost)

        return pheromone_matrice

    def generate_tour_without_drone(self, q0: float) -> tuple[list[int], float, float]:
        """
        Funktion zum Erzeugen einer Tour ohne Drohnenunterstützung
        :param q0: Grenzwert für die Entscheidung zwischen Exploitation und Exploration
        :return: Tupel bestehend aus der Liste der Tour, Distanz der Tour und Dauer der Tour
        """
        # Initialisierung
        stops = range(len(self.coordinates))  # Anzahl der Ziele
        visited_stops = []  # Liste zum Verwalten der besuchten Ziele
        unvisited_stops = set(range(1, len(self.coordinates)))  # Set zum Verwalten der unbesuchten Ziele
        # Ziele aufsteigend sortieren in Abhängigkeit von der Distanz zum Depot
        sorted_stops = sorted(unvisited_stops, key = lambda x: self.distances_truck[0, x])
        loaded_packages = self.capacity_truck  # Ladung des Lieferwagens
        packages_at_depot = self.destinations - loaded_packages  # Lieferungen im Depot
        # Tour, Distanz und Dauer initialisieren
        tour = [0]
        tour_distance = 0
        tour_time = 0

        # Tour planen, solange nicht alle Stops angefahren wurden
        while unvisited_stops:
            # aktuellen Stop ermitteln
            current_stop = tour[-1]
            free_capacity = self.capacity_truck - loaded_packages
            # Heuristische Informationen berechnen
            eta_depot = 0
            eta_load = int(bool(loaded_packages))  # 1 wenn noch Pakete im Lieferwagen sind, sonst 0
            if packages_at_depot:
                eta_depot = min(1.0, max(free_capacity / self.capacity_truck, free_capacity / packages_at_depot))

            eta_best = int(not (bool(packages_at_depot) and  # Pakete im Depot
                                eta_depot == 1 and  # Alle Pakete können mitgenommen werden
                                (current_stop == sorted_stops[0]) and  # Aktueller halt ist dem Depot am nächsten
                                self.force_best))  # Faktor soll genutzt werden
            # Wahrscheinlichkeiten für alle Stops berechnen
            heuristic_information = self.visibility_truck[current_stop] * eta_load * eta_best
            probabilities = ((self.pheromones_truck[current_stop] ** self.alpha) *
                             (heuristic_information ** self.beta))
            # Wahrscheinlichkeit für Fahrt zum Depot berechnen
            heuristic_information = self.visibility_truck[current_stop, 0] * eta_depot
            probabilities[0] = ((self.pheromones_truck[current_stop, 0] ** self.alpha) *
                                (heuristic_information ** self.beta))
            # Wahrscheinlichkeiten für besuchte Stops auf 0 setzen
            np.put(probabilities, visited_stops, 0)
            # Wahrscheinlichkeiten normalisieren
            probabilities = np.array(probabilities / np.sum(probabilities))
            # Zufallswert holen
            q = random.random()
            # nächstes Ziel nach ACS auswählen
            if q < q0:
                next_stop = np.argmax(probabilities)
            else:
                next_stop = np.random.choice(stops, p = probabilities)
            # Ziel zur Tour hinzufügen und Distanz und Dauer aktualisieren
            tour.append(next_stop)
            distance = self.distances_truck[current_stop, next_stop]
            tour_distance += distance
            tour_time += distance / self.velocity_truck
            # lokales Pheromonupdate durchführen
            new_pheromones = self.pheromone_update_local(self.pheromones_truck, current_stop, next_stop)
            self.pheromones_truck[current_stop, next_stop] = new_pheromones
            # Prüfen, ob der nächste Stop ein Lieferziel ist
            if next_stop != 0:
                # Ziel aus der Liste der unbesuchten Ziele entfernen
                unvisited_stops.remove(next_stop)
                if current_stop in sorted_stops:
                    # Ziel aus der Liste mit den sortierten Zielen entfernen
                    sorted_stops.remove(current_stop)
                # Ziel zur Liste der besuchten Ziele hinzufügen
                visited_stops.append(next_stop)
                # Tour-Dauer und Ladung aktualisieren (Auslieferung)
                tour_time += self.delivery_time
                loaded_packages -= 1
            # wenn das nächste Ziel das Depot ist
            else:
                # Berechnen wie viele Pakete nachgeladen werden können
                packages_to_load = min(packages_at_depot, free_capacity)
                # Ladung sowie Lieferungen im Depot entsprechend aktualisieren
                packages_at_depot -= packages_to_load
                loaded_packages += packages_to_load
                # Dauer der Tour aktualisieren (Nachladen)
                tour_time += self.loading_time * packages_to_load

        # Rückfahrt zum Depot hinzufügen
        last_stop = tour[-1]
        tour.append(0)
        # Distanz und Dauer aktualisieren
        distance = self.distances_truck[last_stop, 0]
        tour_distance += distance
        tour_time += distance / self.velocity_truck
        # Generierte Tour, Distanz und Dauer zurückgeben
        return tour, tour_distance, tour_time

    def generate_tour_with_drone(self, q0: float) -> tuple[list[int], float, float, list[int], float, float]:
        """
        Funktion zum Erzeugen einer Liefertour mit Drohnenunterstützung
        :param q0: Grenzwert für die Entscheidung zwischen Exploitation und Exploration
        :return: Tupel mit den Touren, Distanzen und Kosten für Lieferwagen und Drohne
        """
        # Initialisierung
        unvisited_destinations = set(range(1, len(self.coordinates)))  # Set zum Verwalten der unbesuchten Ziele
        # Lieferziele aufsteigend in Abhängigkeit von der Distanz sortieren
        sorted_destinations = sorted(unvisited_destinations, key = lambda x: self.distances_drone[0, x])
        # Tour, Distanz und Dauer für Lieferwagen und Drohne initialisieren
        truck_tour = [0]
        truck_distance = 0
        truck_time = 0
        drone_tour = [0]
        drone_distance = 0
        drone_time = 0
        loaded_packages = self.capacity_truck  # Ladung des Lieferwagens
        packages_at_depot = self.destinations - loaded_packages  # im Depot verbliebene Lieferungen

        # Tour erzeugen, solange nicht alle Ziele besucht wurden
        while unvisited_destinations:
            # aktuellen Stop ermitteln
            current_stop = truck_tour[-1]
            free_capacity = self.capacity_truck - loaded_packages
            # heuristische Informationen berechnen
            eta_load = int(bool(loaded_packages))
            eta_depot = 0
            if packages_at_depot:
                eta_capacity_drone = min(max(free_capacity / self.capacity_drone, free_capacity / packages_at_depot), 1)
                eta_flight_count = 1 / (1 + drone_tour.count(current_stop))
                eta_depot = eta_capacity_drone * eta_flight_count

            eta_best = int(not (current_stop == sorted_destinations[0]
                                and eta_depot == 1
                                and self.force_best))

            # heuristische Information ermitteln
            heuristic_information = self.visibility_truck[current_stop] * eta_load * eta_best
            # Wahrscheinlichkeiten für den Lieferwagen berechnen
            probabilities = np.array((self.pheromones_truck[current_stop] ** self.alpha) *
                                     (heuristic_information ** self.beta))
            # Wahrscheinlichkeiten für besuchte Lieferziele auf 0 setzen
            np.put(probabilities, truck_tour, 0)

            # Wahrscheinlichkeit für die Drohne berechnen
            heuristic_information = (self.visibility_drone[0, current_stop] * eta_depot)
            probabilities[current_stop] = ((self.pheromones_drone[0, current_stop] ** self.alpha) *
                                           (heuristic_information ** self.beta))
            # Wahrscheinlichkeiten normalisieren
            probabilities = probabilities / np.sum(probabilities)
            # Zufallswert holen
            q = random.random()
            # Nächstes Ziel nach ACS wählen
            if q < q0:
                next_stop = np.argmax(probabilities)
            else:
                next_stop = np.random.choice(range(len(self.coordinates)), p = probabilities)
            # Prüfen, ob mit der Drohne nachgeliefert wird
            if next_stop == current_stop:
                # Ladung der Drohne berechnen
                drone_load = min(self.capacity_drone, self.capacity_truck - loaded_packages, packages_at_depot)
                drone_time += self.loading_time * drone_load
                # Ladung von den Paketen im Depot abziehen
                packages_at_depot -= drone_load
                # Flug zum aktuellen Stop zur Drohnen-Tour hinzufügen
                drone_tour.append(current_stop)
                # Distanz berechnen
                distance = self.distances_drone[0, current_stop]
                drone_distance += distance
                # Flugzeit berechnen
                drone_time += distance / self.velocity_drone
                # Ankunftszeit beim Lieferwagen
                truck_time = max(truck_time, drone_time)
                # lokales Pheromon-Update durchführen
                new_pheromones = self.pheromone_update_local(self.pheromones_drone, 0, current_stop)
                self.pheromones_drone[0, current_stop] = new_pheromones
                # Ladung zum Lieferwagen hinzufügen
                loaded_packages += drone_load
                # Ladezeit berechnen
                truck_time += self.loading_time  # * drone_load
                # Rückflug zum Depot zur Drohnen-Tour hinzufügen
                drone_tour.append(0)
                drone_distance += distance
                drone_time = truck_time + distance / self.velocity_drone
            else:  # Andernfalls fährt der Lieferwagen zum nächsten Ziel
                if current_stop in sorted_destinations:
                    # Ziel aus der Liste der sortierten Ziele entfernen
                    sorted_destinations.remove(current_stop)
                # Fahrt zum nächsten Ziel zur Truck-Tour hinzufügen
                truck_tour.append(next_stop)
                # Distanz berechnen
                distance = self.distances_truck[current_stop, next_stop]
                truck_distance += distance
                # Fahrtzeit berechnen
                truck_time += distance / self.velocity_truck
                # lokales Pheromon-Update durchführen
                new_pheromones_truck = self.pheromone_update_local(self.pheromones_truck, current_stop, next_stop)
                self.pheromones_truck[current_stop, next_stop] = new_pheromones_truck
                # Ziel aus der Liste der unbesuchten Ziele entfernen
                unvisited_destinations.remove(next_stop)
                # Lieferung aus der Ladung entfernen
                loaded_packages -= 1
                truck_time += self.delivery_time
        # Rückfahrt zum Depot zur Tour hinzufügen
        last_stop = truck_tour[-1]
        truck_tour.append(0)
        # Distanz und Fahrtzeit berechnen
        distance = self.distances_truck[last_stop, 0]
        truck_distance += distance
        truck_time += distance / self.velocity_truck
        # Einsatzzeit der Drohne berechnen
        drone_time = ((drone_distance / self.velocity_drone) +
                      ((len(drone_tour) - 1) * self.loading_time))

        return truck_tour, truck_distance, truck_time, drone_tour, drone_distance, drone_time

    def run_truck_with_drone_support(self, show_all_plots: bool = True) -> tuple:
        """
        Funktion zum Durchführen der Optimierung mit Drohnenunterstützung
        :param show_all_plots: Parameter zum Steuern, ob jedes Mal ein Plot erzeugt werden sollen,
                                wenn eine bessere Tour gefunden wird. Default = True
        :return: Tupel welches die besten Touren, Distanzen, Kosten, Zeiten und Plot für Lieferwagen und Drohne enthält,
        sowie die Iteration in der das beste Ergebnis gefunden wurde
        """
        # Timer und Iterationen initialisieren
        start = time.perf_counter()
        iteration = 0

        # Nearest neighbour tour erzeugen
        nnt_truck, distance_truck, time_truck, nnt_drone, distance_drone, time_drone = (
            self.generate_tour_with_drone(1))
        # Kosten der NNT berechnen
        nnt_cost = (distance_truck * self.cost_per_km_truck +
                    distance_drone * self.cost_per_km_drone +
                    time_truck * self.cost_per_h_truck +
                    time_drone * self.cost_per_h_drone)

        # tau0 basieren auf den Kosten der NNT berechnen und Pheromone reinitialisieren
        self.tau_0 = 1 / (self.destinations * nnt_cost)
        self.reinit_pheromones()

        # Globals initialisieren
        global_best_tour = None
        global_best_distance = np.inf
        global_best_time = np.inf
        global_best_cost = np.inf
        best_iteration = 0

        # Namen und Plot-Stil der Touren definieren
        names = ['Truck:', 'Drone:']
        styles = ['o-', 'x:']

        # Optimierung ausführen, solange max_duration oder max_iterations nicht überschritten
        while (time.perf_counter() - start < self.max_duration) & (iteration < self.max_iterations):
            iteration += 1
            # Listen zum Verwalten der erzeugten Touren, Distanzen, Kosten und Zeiten in der Iteration
            tours = []
            distances = []
            costs = []
            times = []

            # Schleife für die Ameisen zur Generierung der Touren
            for _ in range(self.n_ants):
                # Tour generieren
                tour_truck, distance_truck, time_truck, tour_drone, distance_drone, time_drone = (
                    self.generate_tour_with_drone(self.q0))
                if self.use_min_max:
                    np.clip(self.pheromones_truck, self.tau_min, self.tau_0)
                    np.clip(self.pheromones_drone, self.tau_min, self.tau_0)
                # Tour, Distanz und Dauer speichern
                tours.append((tour_truck, tour_drone))
                total_distance = distance_truck + distance_drone
                distances.append((distance_truck, distance_drone, total_distance))
                times.append((time_truck, time_drone))
                # Kosten berechnen und
                tour_costs = (distance_truck * self.cost_per_km_truck +
                              distance_drone * self.cost_per_km_drone +
                              time_truck * self.cost_per_h_truck +
                              time_drone * self.cost_per_h_truck)
                costs.append(tour_costs)

            # Beste Tour basierend auf den Kosten ermitteln
            min_index = np.argmin(costs)
            best_tour = tours[min_index]
            best_distance = distances[min_index]
            best_time = times[min_index]
            best_cost = costs[min_index]
            # Global beste Tour überschreiben, falls die beste Tour aus der aktuellen Iteration besser ist
            if best_cost < global_best_cost:
                best_iteration = iteration
                global_best_distance = best_distance
                global_best_tour = best_tour
                global_best_cost = best_cost
                global_best_time = best_time

                runtime = round(time.perf_counter() - start, 2)
                est_time = round(min(runtime / iteration * self.max_iterations, self.max_duration), 2)
                print(f'Iteration {iteration} - Runtime: {runtime}s - Est. Runtime: {est_time}s')
                print(f'Best tour: {global_best_tour}\n'
                      f'Best distance: {global_best_distance}\n'
                      f'Best time: {global_best_time}\n'
                      f'Best cost: {global_best_cost}')
                if show_all_plots:
                    # Touren plotten
                    fig = self.plot_tours(global_best_tour,
                                          global_best_distance,
                                          global_best_time,
                                          global_best_cost,
                                          names,
                                          styles,
                                          True)
                    # Plot anzeigen
                    fig.show()
            # Globales Pheromon-Update durchführen
            self.pheromones_truck = self.pheromone_update_global(self.pheromones_truck, global_best_tour[0],
                                                                 global_best_cost)
            self.pheromones_drone = self.pheromone_update_global(self.pheromones_drone, global_best_tour[1],
                                                                 global_best_cost)
            if self.use_min_max:
                # Min-Max AS anwenden
                self.tau_0 = 1 / (self.destinations * global_best_cost)
                np.clip(self.pheromones_truck, self.tau_min, self.tau_0)
                np.clip(self.pheromones_drone, self.tau_min, self.tau_0)

        # Tour-Dauer runden
        durations = [round(duration, 1) for duration in global_best_time]
        # Plot erstellen
        fig = self.plot_tours(global_best_tour,
                              global_best_distance,
                              durations,
                              global_best_cost,
                              names,
                              styles,
                              True)
        # fig.show()
        return global_best_tour, global_best_distance, global_best_time, global_best_cost, best_iteration, fig

    def run_truck_without_drone_support(self, show_all_plots: bool = True) -> tuple:
        """
        Funktion zum Durchführen der Optimierung von Liefertouren ohne Drohnenunterstützung
        :param show_all_plots: Parameter zum Steuern, ob jedes Mal ein Plot erzeugt werden sollen,
                                wenn eine bessere Tour gefunden wird. Default = True
        :return: Tupel welches die beste Tour, Distanz, Kosten, Zeit und Plot für den Lieferwagen enthält,
        sowie die Iteration in der das beste Ergebnis gefunden wurde
        """
        # Timer und Iterationen initialisieren
        start = time.perf_counter()
        iteration = 0

        # Nearest Neighbor Tour erzeugen und Kosten berechnen
        nnt, nnt_distance, nnt_time = self.generate_tour_without_drone(1)
        nnt_cost = nnt_distance * self.cost_per_km_truck + nnt_time * self.cost_per_h_truck

        # Pheromon-Matrix basierend auf den Kosten der NNT reinitialisieren
        self.tau_0 = 1 / (self.destinations * nnt_cost)
        self.reinit_pheromones()

        # Global beste Tour, Distanz, Kosten und Dauer initialisieren
        best_iteration = 0
        global_best_tour = None
        global_best_distance = np.inf
        global_best_cost = np.inf
        global_best_time = np.inf

        # Ausführen, solange max_duration oder max_iterations nicht überschritten
        while (time.perf_counter() - start < self.max_duration) & (iteration < self.max_iterations):
            iteration += 1
            # Listen zum Verwalten der generierten Touren initialisieren
            tours = []
            distances = []
            costs = []
            times = []

            # Schleife für die Ameisen zur Generierung der LKW-Touren
            for _ in range(self.n_ants):
                # LKW-Tour generieren
                tour, tour_distance, tour_time = self.generate_tour_without_drone(self.q0)
                if self.use_min_max:
                    # Min-Max AS anwenden
                    np.clip(self.pheromones_truck, self.tau_min, self.tau_0)
                    np.clip(self.pheromones_drone, self.tau_min, self.tau_0)
                # Tour, Distanz, Dauer und Kosten speichern
                tours.append(tour)
                distances.append(tour_distance)
                times.append(tour_time)
                cost = tour_distance * self.cost_per_km_truck + tour_time * self.cost_per_h_truck
                costs.append(cost)

            # Beste Tour der Iteration ermitteln
            argmin = np.argmin(costs)
            best_tour = tours[argmin]
            best_distance = distances[argmin]
            best_cost = costs[argmin]
            best_time = times[argmin]

            # Global beste Tour durch beste Tour der Iteration ersetzen, wenn diese besser ist
            if best_cost < global_best_cost:
                best_iteration = iteration
                global_best_tour = best_tour
                global_best_distance = best_distance
                global_best_cost = best_cost
                global_best_time = best_time

                # aktuelle und geschätzte Laufzeit berechnen
                runtime = round(time.perf_counter() - start, 2)
                est_time = round(min(runtime / iteration * self.max_iterations, self.max_duration), 2)

                # Informationen in der Konsole ausgeben
                print(f'Iteration {iteration} - Runtime: {runtime}s - Est. Runtime: {est_time}s')
                print(f'Best tour: {global_best_tour}\n'
                      f'Best distance: {global_best_distance}\n'
                      f'Best time: {global_best_time}\n'
                      f'Best cost: {global_best_cost}')

                if show_all_plots:
                    # Einzeltouren finden
                    split_tours, names, styles = split_tour(global_best_tour)
                    # Distanzen der Einzeltouren berechnen
                    split_distances = [calculate_tour_distance(tour, self.distances_truck) for tour in split_tours]
                    split_distances.append(global_best_distance)
                    # Touren plotten und anzeigen
                    fig = self.plot_tours(split_tours,
                                          split_distances,
                                          global_best_time,
                                          global_best_cost,
                                          names,
                                          styles,
                                          False)
                    fig.show()
            # globales Pheromonupdate durchführen
            self.pheromones_truck = self.pheromone_update_global(self.pheromones_truck,
                                                                 global_best_tour,
                                                                 global_best_cost)
            if self.use_min_max:
                # Min-Max AS anwenden
                self.tau_0 = 1 / (self.destinations * global_best_cost)
                np.clip(self.pheromones_truck, self.tau_min, self.tau_0)
                np.clip(self.pheromones_drone, self.tau_min, self.tau_0)

        # Einzeltouren der besten gefundenen Tour ermitteln
        split_tours, names, styles = split_tour(global_best_tour)
        split_distances = [calculate_tour_distance(tour, self.distances_truck) for tour in split_tours]
        split_distances.append(global_best_distance)
        # Touren plotten
        fig = self.plot_tours(split_tours,
                              split_distances,
                              global_best_time,
                              global_best_cost,
                              names,
                              styles,
                              False)
        # Ergebnisse zurückgeben
        return global_best_tour, global_best_distance, global_best_time, global_best_cost, best_iteration, fig

    def plot_tours(self,
                   tours: list[list[int]] | tuple[list[int], list[int]],
                   distances: list[float] | tuple[float, float, float],
                   durations: float | list[float] | tuple[float, float],
                   costs: list[float] | float,
                   names: list[str],
                   styles: list[str],
                   with_drone: bool) -> plt.Figure:
        """
        Funktion zum Plotten von Touren
        :param tours: Listen der Touren die geplottet werden sollen
        :param distances: Liste mit den Distanzen der einzelnen Touren sowie der Gesamtdistanz
        :param durations: Dauer der Touren
        :param costs: Kosten der Touren
        :param names: Bezeichnung der Touren
        :param styles: Plot-Stil der Touren
        :param with_drone: Parameter, der steuert, ob eine Drohnentour geplottet werden soll
        :return: Plot mit den Touren
        """
        # Figure mit zwei Subplots erzeugen für die Touren und Legende
        fig, ax = plt.subplots(ncols = 2, gridspec_kw = {'width_ratios': [5, 2]}, figsize = (7, 5),
                               layout = 'constrained')

        for i in range(len(tours)):
            # Länge der Tour ermitteln
            stops = len(tours[i]) - tours[i].count(0)
            # Tour auf erstem Subplot plotten
            ax[0].plot(self.coordinates[tours[i], 0], self.coordinates[tours[i], 1], styles[i],
                       label = f'{names[i]} {round(distances[i], 1)}km ({stops})')
            for j in range(len(tours[i])):
                if tours[i][j] != 0 and names[i] != 'Drone:':
                    # Reihenfolge, in welcher die Ziele besucht werden annotieren, wenn es sich nicht um eine
                    # Drohnentour handelt
                    ax[0].text(self.coordinates[tours[i][j], 0] + 0.15, self.coordinates[tours[i][j], 1] + 0.15, f'{j}')
        handles, labels = ax[0].get_legend_handles_labels()
        # Text mit zusätzlichen Informationen über die Tour erzeugen
        text = (f'Total Distance: {round(distances[-1], 1)}km\n'
                f'Costs: {round(costs, 1)}\n'
                f'Capacity Truck: {self.capacity_truck}\n')
        if with_drone:
            duration_truck = round(durations[0], 1)
            duration_drone = round(durations[1], 1)
            text += (f'Duration Truck: {duration_truck}h\n'
                     f'Capacity Drone: {self.capacity_drone}\n'
                     f'Duration Drone: {duration_drone}h\n')
        else:
            text += f'Duration: {round(durations, 1)}h\n'
        # Informationen und Legende auf dem zweiten Subplot platzieren
        ax[1].text(0, 0, text)
        ax[1].legend(handles, labels)
        # Achsen für den zweiten Subplot entfernen
        ax[1].axis('off')

        # Titel festlegen
        title = 'Tour with drone support' if with_drone else 'Tour without drone support'
        fig.suptitle(title)

        return fig
