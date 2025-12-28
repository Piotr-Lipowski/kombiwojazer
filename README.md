# Problem Komiwojażera - Algorytm Genetyczny

Projekt implementuje algorytm genetyczny rozwiązujący problem komiwojażera (TSP) z wizualizacją w Streamlit.

## Funkcje
- Wybór datasetów: ATT48, Berlin52.
- Parametryzacja GA: rozmiar populacji, liczba generacji, prawdopodobieństwa mutacji i krzyżowania, elitaryzm.
- Różne operatory:
    - Selekcja: Turniejowa, Ruletka.
    - Krzyżowanie: OX, PMX.
    - Mutacja: Swap, Inversion.
- Wizualizacja na żywo najlepszej trasy i wykresu zbieżności.
- Eksport wyników do JSON i CSV.

## Instalacja
```bash
pip install -r requirements.txt
```

## Uruchomienie
```bash
streamlit run app.py
```

## Struktura plików
- `app.py`: Główna aplikacja Streamlit.
- `genetic_algorithm.py`: Logika algorytmu genetycznego.
- `tsp_problem.py`: Obsługa wczytywania danych i obliczania odległości.
- `operators.py`: Implementacja operatorów genetycznych.
- `visualization.py`: Funkcje rysujące wykresy.
- `data/`: Pliki z danymi TSP.
- `results/`: Wyniki eksperymentów.
