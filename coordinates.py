from random import choice
import osmnx as ox
import networkx as nx

# Straßennetz für eine bestimmte Stadt abrufen
G = ox.graph_from_place("Redmond, Washington, USA", network_type = "all")

# Zufälligen Straßenknoten auswählen
random_node = choice(list(G.nodes()))

# Koordinaten des Knotens abrufen
x, y = G.nodes[random_node]["x"], G.nodes[random_node]["y"]
print(f"Zufällige Koordinaten auf dem Straßennetz: x={x}, y={y}")

# Straßennetz für eine bestimmte Stadt abrufen
G = ox.graph_from_place("Redmond, Washington, USA", network_type = "all")

# Start- und Endknoten auswählen (z. B. durch Koordinaten)
start_coords = (47.6740, -122.1215)
end_coords = (47.6781, -122.1306)

# Die nächsten Straßenknoten zu den Koordinaten finden
start_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
end_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])

# Kürzesten Pfad (Straßendistanz) zwischen den Knoten berechnen
try:
    path = nx.shortest_path(G, start_node, end_node, weight = "length")
    street_distance = sum(G[u][v].get("length", 0) for u, v in zip(path[:-1], path[1:]))
    print(f"Straßendistanz zwischen den Koordinaten: {street_distance:.2f} Meter")
except nx.NetworkXNoPath:
    print("Kein gültiger Straßenpfad zwischen den Koordinaten gefunden.")
