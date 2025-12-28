import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt
from tsp_problem import TSPProblem
from genetic_algorithm import GeneticAlgorithm
from visualization import plot_route, plot_convergence

# Konfiguracja strony
st.set_page_config(page_title="TSP Genetic Algorithm", layout="wide")

st.title("Rozwiązywanie problemu komiwojażera (TSP) algorytmem genetycznym")

# Sidebar - Parametry
st.sidebar.header("Parametry Algorytmu")

dataset_name = st.sidebar.selectbox("Dataset", ["ATT48", "Berlin52"])
pop_size = st.sidebar.slider("Rozmiar populacji", 50, 500, 100)
num_generations = st.sidebar.slider("Liczba generacji", 100, 2000, 500)
mutation_rate = st.sidebar.slider("Prawdopodobieństwo mutacji", 0.01, 0.2, 0.05)
crossover_rate = st.sidebar.slider("Prawdopodobieństwo krzyżowania", 0.5, 1.0, 0.8)
elite_size = st.sidebar.slider("Rozmiar elity (%)", 0, 20, 10) / 100.0

selection_method = st.sidebar.selectbox("Metoda selekcji", ["Tournament", "Roulette Wheel"])
crossover_method = st.sidebar.selectbox("Metoda krzyżowania", ["OX", "PMX"])
mutation_method = st.sidebar.selectbox("Metoda mutacji", ["Swap", "Inversion"])

start_button = st.sidebar.button("START")

# Mapowanie datasetów na pliki i optima
datasets = {
    "ATT48": {"file": "data/att48.tsp", "optimum": 10628},
    "Berlin52": {"file": "data/berlin52.tsp", "optimum": 7542}
}

if start_button:
    file_path = datasets[dataset_name]["file"]
    optimum = datasets[dataset_name]["optimum"]
    
    if not os.path.exists(file_path):
        st.error(f"Nie znaleziono pliku {file_path}. Pobierz dane przed uruchomieniem.")
    else:
        # Inicjalizacja problemu i algorytmu
        tsp = TSPProblem(file_path)
        ga = GeneticAlgorithm(
            tsp, 
            population_size=pop_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_size=elite_size,
            selection_method=selection_method,
            crossover_method=crossover_method,
            mutation_method=mutation_method
        )
        
        # UI - Główne kolumny i metryki
        col1, col2 = st.columns(2)
        with col1:
            map_placeholder = st.empty()
        with col2:
            chart_placeholder = st.empty()
            
        metrics_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        # Pętla ewolucji
        for gen in range(1, num_generations + 1):
            best_route, best_dist = ga.evolve()
            
            # Aktualizacja co 20 generacji lub na końcu
            if gen % 20 == 0 or gen == num_generations:
                # Metryki
                diff_opt = ((best_dist - optimum) / optimum) * 100
                improvement = ((ga.history[0] - best_dist) / ga.history[0]) * 100 if ga.history else 0
                
                with metrics_placeholder.container():
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Najlepszy dystans", f"{int(best_dist)}")
                    m2.metric("Różnica od optimum", f"{diff_opt:.2f}%")
                    m3.metric("Generacja", f"{gen}/{num_generations}")
                    m4.metric("Czas", f"{time.time() - start_time:.2f}s")
                    m5.metric("Poprawa", f"{improvement:.2f}%")
                
                # Wykresy
                fig_route = plot_route(tsp, best_route)
                map_placeholder.pyplot(fig_route)
                plt.close(fig_route)
                
                fig_conv = plot_convergence(ga.history, optimum)
                chart_placeholder.pyplot(fig_conv)
                plt.close(fig_conv)
                
                progress_bar.progress(gen / num_generations)
                status_text.text(f"Ewolucja w toku... Generacja {gen}")
                
        end_time = time.time()
        status_text.success(f"Zakończono! Całkowity czas: {end_time - start_time:.2f}s")
        
        # Eksport wyników
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        with open(f"{results_dir}/best_route.json", "w") as f:
            json.dump({"route": [int(i) for i in best_route], "distance": float(best_dist)}, f)
        
        # Logowanie eksperymentu
        exp_data = {
            "Dataset": dataset_name,
            "PopSize": pop_size,
            "Generations": num_generations,
            "MutRate": mutation_rate,
            "CrossRate": crossover_rate,
            "EliteSize": elite_size,
            "Selection": selection_method,
            "Crossover": crossover_method,
            "Mutation": mutation_method,
            "BestDist": best_dist,
            "DiffOpt": diff_opt,
            "Time": end_time - start_time
        }
        
        csv_path = f"{results_dir}/experiments.csv"
        df = pd.DataFrame([exp_data])
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
            
else:
    st.info("Dostosuj parametry w pasku bocznym i naciśnij START, aby rozpocząć algorytm.")
