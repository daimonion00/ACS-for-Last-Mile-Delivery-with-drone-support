"""
Author: Georg Börries
Last Updated: 24.04.2024

Hauptmodul für die Optimierung der Last Mile Delivery mit Drohnenunterstützung
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd

import utilities
from ant_colony_system import AntColonySystem


def main():
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M")
    # Parameter für Koordinaten festlegen
    num_coordinates = 50  # Anzahl der zu generierenden Lieferpunkte
    max_distance = 5  # Maximaler Abstand in km
    depot = (0, 0)  # Koordinaten Depot

    # Koordinaten erzeugen
    delivery_coordinates = np.array(utilities.generate_delivery_coordinates(depot, num_coordinates, max_distance))

    # Listen für die Ergebnisse initialisieren
    results = []
    # ACS Parameter
    max_iterations = 3000
    max_duration = 600
    alpha = 1
    beta = 2
    ants = 100
    rho = 0.075
    q0 = 0.85

    capacity_truck = 45
    capacity_drone = 2
    velocity_truck = 25
    velocity_drone = 50
    cost_per_km_truck = 0.7
    cost_per_km_drone = 0.35
    cost_per_h_truck = 15
    cost_per_h_drone = 20
    loading_time = 0.5 / 60
    delivery_time = 3 / 60

    folder = f'./tests/drone_capacity_{capacity_drone}/{max_distance}km/{num_coordinates}coordinates/{date_str}/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # ACS initialisieren
    acs = AntColonySystem(delivery_coordinates,
                          ants,
                          alpha,
                          beta,
                          rho,
                          q0,
                          max_iterations,
                          max_duration,
                          capacity_truck,
                          capacity_drone,
                          velocity_truck,
                          velocity_drone,
                          cost_per_km_truck,
                          cost_per_km_drone,
                          cost_per_h_truck,
                          cost_per_h_drone,
                          loading_time,
                          delivery_time)
    # Tour ohne Drohnen-Support optimieren
    # tour, distance, time, cost, best_iteration, fig = acs.run_truck_without_drone_support()
    # fig.savefig(f'{folder}{n}-tour_without_drone-{capacity}_{capacity_drone}.png')
    # results.append((ants, alpha, beta, rho, q0,
    #                 tour,
    #                 capacity, 0,
    #                 distance, 0,
    #                 time, 0,
    #                 cost))
    # Pheromone zurücksetzen
    acs.reset_pheromones()
    # Tour mit Drohnen-Support optimieren
    tour_drone, distance_drone, time_drone, cost_drone, best_iteration, fig = acs.run_truck_with_drone_support()
    # fig.savefig(f'{folder}{n}-tour_with_drone-{capacity}_{capacity_drone}.png')
    # results.append((ants, alpha, beta, rho, q0,
    #                 tour, tour_drone,
    #                 capacity_truck, capacity_drone,
    #                 distance, distance_drone[0], distance_drone[1],
    #                 time, time_drone[0], time_drone[1],
    #                 cost, cost_drone))
    # utilities.plot_graphs(distance,distance_drone,time,time_drone)
    # plt.close('all')

    df_columns = ['ants', 'alpha', 'beta', 'rho', 'q0',
                  'tour', 'tour_drone',
                  'capacity_truck', 'capacity_drone',
                  'distance_truck', 'distance_truck', 'distance_drone',
                  'time_truck', 'time_truck', 'time_drone',
                  'cost', 'cost_drone']
    df_results = pd.DataFrame(results, columns = df_columns)

    # # Koordinaten speichern
    # np.save(f'{folder}coordinates.npy', delivery_coordinates)
    # df_results.to_excel(f"{folder}results.xlsx", float_format = '%.2f')


if __name__ == "__main__":
    main()
