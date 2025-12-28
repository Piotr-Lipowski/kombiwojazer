import matplotlib.pyplot as plt
import numpy as np

def plot_route(tsp_problem, route, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
    coords = tsp_problem.coords
    route_coords = [coords[i] for i in route]
    route_coords.append(coords[route[0]]) # Close the loop
    
    xs, ys = zip(*route_coords)
    
    ax.plot(xs, ys, 'o-', markersize=5, linewidth=1, color='blue')
    ax.plot(xs[0], ys[0], 'rs', markersize=8, label='Start')
    ax.set_title(f"Najlepsza trasa (Dystans: {tsp_problem.get_total_distance(route)})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    return fig

def plot_convergence(history, optimum=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
    ax.plot(history, color='green', linewidth=2)
    if optimum:
        ax.axhline(y=optimum, color='red', linestyle='--', label=f'Optimum ({optimum})')
        ax.legend()
        
    ax.set_title("Zbieżność algorytmu")
    ax.set_xlabel("Generacja")
    ax.set_ylabel("Dystans")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    return fig
