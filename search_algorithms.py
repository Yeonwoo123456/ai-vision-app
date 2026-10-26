"""Graph search utilities for AI Vision Lab."""

import heapq
import itertools
import math

import numpy as np


def _landmark_vector(point):
    values = [float(point.get(axis, 0.0)) for axis in ("x", "y", "z")]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Landmark coordinates must be finite numbers.")
    return np.asarray(values, dtype=np.float64)


def astar_landmark_path(landmarks, start, goal, k_neighbors=6):
    """Find a minimum-cost route on a k-nearest-neighbor landmark graph using A*.

    Edge costs and the Euclidean heuristic use the same normalized 3D coordinate
    metric. The heuristic is therefore admissible and consistent for this graph.
    """
    count = len(landmarks)
    if count < 2:
        raise ValueError("At least two landmarks are required.")
    if not 0 <= int(start) < count or not 0 <= int(goal) < count:
        raise IndexError("Start and goal must refer to existing landmarks.")
    if int(start) == int(goal):
        return {"path": [int(start)], "cost": 0.0, "expanded_nodes": 0}

    vectors = np.vstack([_landmark_vector(point) for point in landmarks])
    differences = vectors[:, None, :] - vectors[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    neighbors = max(1, min(int(k_neighbors), count - 1))

    graph = [set() for _ in range(count)]
    for node in range(count):
        nearest = np.argpartition(distances[node], neighbors)[:neighbors + 1]
        nearest = [int(other) for other in nearest if int(other) != node]
        nearest.sort(key=lambda other: distances[node, other])
        for other in nearest[:neighbors]:
            graph[node].add(other)
            graph[other].add(node)

    start, goal = int(start), int(goal)
    def heuristic(node):
        return float(np.linalg.norm(vectors[node] - vectors[goal]))

    counter = itertools.count()
    frontier = [(heuristic(start), next(counter), start)]
    came_from = {}
    g_score = {start: 0.0}
    closed = set()
    expanded_nodes = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in closed:
            continue
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return {
                "path": path,
                "cost": float(g_score[goal]),
                "expanded_nodes": expanded_nodes,
            }

        closed.add(current)
        expanded_nodes += 1
        for neighbor in graph[current]:
            tentative_g = g_score[current] + float(distances[current, neighbor])
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                if neighbor in closed:
                    closed.remove(neighbor)
                heapq.heappush(
                    frontier,
                    (tentative_g + heuristic(neighbor), next(counter), neighbor),
                )

    raise ValueError("No path was found in the generated landmark graph.")
