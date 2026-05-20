# DnoOka — segmentacja naczyń dna oka

Binarna segmentacja naczyń krwionośnych na obrazach dna siatkówki oka.

Realizacja etapowa — trzy poziomy trudności odpowiadające ocenom 3, 4 i 5:

1. **Etap 1** — klasyczne przetwarzanie obrazu: kanał zielony -> CLAHE -> filtr Frangiego -> morfologia.
2. **Etap 2** — klasyczny ML: patche 5×5 + cechy (momenty Hu, statystyki) + Random Forest.
3. **Etap 3** — deep learning: U-Net na PyTorch.
4. **Etap 4** — interaktywna aplikacja (`app.ipynb`) + porównanie trzech metod.
5. **Etap 5** — finalny raport (`report/raport.md`).

## Instalacja

```bash
# 1. Utworzyć wirtualne środowisko
python3 -m venv .venv
source .venv/bin/activate

# 2. Zainstalować zależności
pip install -r requirements.txt
```

## Pobranie danych (HRF)

Używana jest baza **HRF Image Database**: https://www5.cs.fau.de/research/data/fundus-images/

1. Na stronie bazy znaleźć sekcję _Segmentation Dataset_ i pobrać archiwum **`all.zip`** (~73 MB). Zawiera wszystkie 45 obrazów wraz z maskami eksperckimi naczyń oraz maskami FOV.
2. Rozpakować do `data/HRF/` tak, aby uzyskać następującą strukturę:

```
data/HRF/
├── images/        # 45 JPG: 01_h.jpg, ..., 15_h.jpg, 01_dr.JPG, ..., 15_g.jpg
├── manual1/       # 45 masek eksperckich naczyń (TIFF/PNG, 0/255)
└── mask/          # 45 masek FOV (TIFF/PNG, 0/255)
```

Pliki nazwane są: `<NN>_<typ>`, gdzie typ = `h` (healthy), `dr` (diabetic retinopathy), `g` (glaucoma). Np.: `01_h`, `07_dr`, `15_g`.

## Uruchomienie

Wszystkie notebooki uruchamiane są z korzenia repozytorium, żeby względne ścieżki `data/...`, `src/...`, `models/...` działały poprawnie.

Następnie po kolei:

1. `notebooks/01_baseline.ipynb` — etap 1
2. `notebooks/02_classical_ml.ipynb` — etap 2 (zapisuje `models/rf.joblib`)
3. `notebooks/03_deep_learning.ipynb` — etap 3 (zapisuje `models/unet.pt`)
4. `notebooks/app.ipynb` — finalna aplikacja i porównanie

## Struktura

```
DnoOka/
├── requirements.txt
├── README.md
├── data/HRF/                       # dane wejściowe (pobrać ręcznie)
├── notebooks/                      # notebooki rozwojowe i aplikacja
├── src/                            # moduły wielokrotnego użytku
│   ├── config.py                   # ścieżki i stałe
│   ├── io_utils.py                 # wczytywanie obrazów
│   ├── preprocessing.py            # CLAHE, kanał zielony
│   ├── baseline.py                 # Frangi + morfologia
│   ├── features.py                 # cechy patchy
│   ├── classical_ml.py             # pipeline Random Forest
│   ├── unet.py                     # architektura U-Net + train/predict
│   ├── metrics.py                  # accuracy, sensitivity, specificity, G-mean
│   └── visualize.py                # overlay i siatki porównawcze
├── models/                         # zapisane wagi (rf.joblib, unet.pt)
├── results/                        # wyniki JSON dla każdego etapu
└── report/                         # finalny raport
```
