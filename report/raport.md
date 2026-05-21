# Wykrywanie naczyń dna siatkówki oka - raport

**Przedmiot:** Informatyka w Medycynie (IwM)
**Projekt:** projekt 2 (DnoOka) - segmentacja naczyń krwionośnych dna oka

---

## 1. Skład grupy

- _Imię i nazwisko 1_
- _Imię i nazwisko 2_

---

## 2. Zastosowany język programowania oraz dodatkowe biblioteki

**Język:** Python 3.10+

**Aplikacja:** Jupyter Notebook - cztery notebooki w katalogu `notebooks/`:

1. `01_baseline.ipynb` - przetwarzanie obrazu (Frangi + morfologia), ocena 3.
2. `02_classical_ml.ipynb` - Random Forest na patchach 5×5, ocena 4.
3. `03_deep_learning.ipynb` - U-Net na PyTorch, ocena 5.
4. `app.ipynb` - interaktywna aplikacja i porównanie trzech metod.

Wspólny kod wydzielony do modułów `src/*.py`.

**Biblioteki** (pełna lista: `requirements.txt`):

- `numpy`, `scipy` - operacje na tablicach.
- `scikit-image` - filtr Frangiego, próg Otsu, operacje morfologiczne.
- `opencv-python` - CLAHE, momenty Hu, gradient Sobela.
- `Pillow` - wczytywanie obrazów i masek (JPG, TIFF, PNG).
- `scikit-learn` - `RandomForestClassifier`, `train_test_split`, `confusion_matrix`.
- `imbalanced-learn` - `RandomUnderSampler`, pipeline z fit_resample.
- `torch` + `torchvision` - implementacja i trening U-Net.
- `joblib` - serializacja modelu Random Forest.
- `matplotlib`, `pandas`, `ipywidgets`, `tqdm` - wizualizacja, tabele, widgety, pasek postępu.

**Baza obrazów:** HRF Image Database (<https://www5.cs.fau.de/research/data/fundus-images/>) - 45 obrazów (15 zdrowych, 15 z retinopatią cukrzycową, 15 z jaskrą), wraz z eksperckimi maskami naczyń (`manual1/`) i maskami FOV (`mask/`). Ta sama baza używana we wszystkich etapach.

**Podział danych (hold-out, ustalony w `src/config.py`):**

- **Train:** 39 obrazów (po 13 z każdej kategorii).
- **Test:** 6 obrazów: `14_h, 15_h, 14_dr, 15_dr, 14_g, 15_g` - wspólnych dla wszystkich trzech metod.

---

## 3. Opis zastosowanych metod

### 3.1. Przetwarzanie obrazów (etap 1, ocena 3)

#### Kroki przetwarzania

1. **Wstępne przetwarzanie** (`src/preprocessing.py`)

   - Wyodrębnienie kanału zielonego - naczynia mają na nim największy kontrast.
   - Wypełnienie pikseli poza FOV medianą wewnątrz FOV - usuwa ostrą krawędź ramki, żeby Frangi nie traktował jej jako naczynia.
   - CLAHE (`cv2.createCLAHE`, `clipLimit=2.0`, `tileGridSize=(8, 8)`) - lokalne wyrównanie kontrastu z ograniczeniem wzmocnienia szumu.

2. **Właściwe przetwarzanie** (`src/baseline.py`)

   - Filtr Frangiego (`skimage.filters.frangi`, `sigmas=(1,2,3,4,5)`, `black_ridges=True`) - detektor struktur tubularnych; na kanale zielonym naczynia są ciemniejsze od tła.
   - Opcjonalny downsampling przed filtrem (`scale=0.5`) - przyspiesza ~4×.
   - Normalizacja odpowiedzi do [0, 1].
   - Próg Otsu (`skimage.filters.threshold_otsu`) liczony **tylko po pikselach wewnątrz FOV** - inaczej czarna ramka przesuwa próg w stronę zera.

3. **Końcowe przetwarzanie**
   - Usunięcie małych komponentów (`skimage.morphology.remove_small_objects`, `min_size=60`).
   - Zamknięcie morfologiczne dyskiem o promieniu 1 (`skimage.morphology.binary_closing`) - łączy drobne przerwy w naczyniach.
   - Końcowe obcięcie maski do FOV.

#### Uzasadnienie

Filtr Frangiego to klasyczny detektor naczyń o dowolnej orientacji, oparty na analizie hesjanu obrazu w wielu skalach. W połączeniu z CLAHE (wyrównanie kontrastu) i progowaniem Otsu (bez parametrów do ręcznego dobrania) daje sensowny baseline bez uczenia. Operacje morfologiczne dodatkowo czyszczą wynik z drobnego szumu.

### 3.2. Klasyczne uczenie maszynowe (etap 2, ocena 4)

#### Przygotowanie danych - wycinki i ekstrakcja cech

- Z każdego obrazu uczącego wyznaczamy patche **5×5 px** wokół każdego piksela.
- Etykieta patcha = wartość maski eksperckiej w **środkowym pikselu**.
- Sampling zbalansowany (~50% naczynia / 50% tło) wewnątrz FOV po `N_SAMPLES_PER_IMAGE = 5000` punktów na obraz; pełny dataset budowany w `src/features.build_dataset`.

Wektor cech (24 wartości, `src/features._features_from_batches`):

- 6 statystyk RGB - średnia i wariancja w każdym z trzech kanałów.
- 2 cechy „środkowego piksela" - kanał G surowy i po CLAHE.
- 4 statystyki patcha na kanale G - mean, var, min, max.
- **7 momentów Hu** (`cv2.HuMoments`) z patcha binaryzowanego progiem średniej jasności.
- 3 momenty centralne - `mu20`, `mu02`, `mu11` (`cv2.moments`).
- 2 statystyki magnitudy gradientu Sobela - mean i var.

#### Wstępne przetwarzanie zbioru uczącego

- Stratyfikowany podział train / val 2/3 : 1/3 (`sklearn.model_selection.train_test_split`).
- Balansowanie klas przez **undersampling** (`imblearn.under_sampling.RandomUnderSampler`, `sampling_strategy='auto'`) - wyrównanie do 50/50.

#### Klasyfikator i parametry

- **Random Forest** (`sklearn.ensemble.RandomForestClassifier`):
  - `n_estimators = 100`
  - `max_depth = None` (bez ograniczenia)
  - `min_samples_leaf = 2`
  - `random_state = 42`
- Pipeline `imblearn.Pipeline` (`undersample -> rf`) - resampling stosowany **tylko podczas fit**.

#### Wyniki hold-out (z notebooka `02_classical_ml.ipynb`)

> _Wpisać liczby z komórki „Validation" notebooka (sekcja 5):_

| accuracy     | sensitivity  | specificity  | g_mean       |
| ------------ | ------------ | ------------ | ------------ |
| \_0.\_\_\_\_ | \_0.\_\_\_\_ | \_0.\_\_\_\_ | \_0.\_\_\_\_ |

#### Uzasadnienie

Random Forest został wybrany jako odporny na nieliniowe zależności i nieskorelowany z większością cech jednocześnie (vs. SVM, który wymagałby skalowania). Hand-crafted cechy łączą informację koloru (statystyki RGB/G), kształtu (momenty Hu - niezmiennicze względem skali i obrotu) i krawędzi (Sobel) - dzięki temu model „widzi" patch w kilku komplementarnych reprezentacjach.

### 3.3. Głębokie uczenie (etap 3, ocena 5)

#### Architektura

- **U-Net** zaimplementowany w PyTorch (`src/unet.py`): 4 poziomy enkodera/dekodera, `base_channels = 32`, łącznie ~4 mln parametrów.
- Wejście: 3 kanały (RGB). Wyjście: 1 kanał (logit prawdopodobieństwa naczynia).

#### Przygotowanie danych

- Wewnątrz puli train (39 obrazów) wydzielony inner-val (6 obrazów) - kontrola overfittingu w trakcie uczenia. `TEST_IMAGES` pozostaje nietknięty do końcowej oceny.
- Z każdego obrazu losowe kropy **256×256** wewnątrz FOV (`crops_per_train_image = 64`).
- Sampling środków kropów z biasem na naczynia (`vessel_prob = 0.5`).
- Augmentacje: obroty o k·90°, odbicie poziome i pionowe.
- Normalizacja per-kanał statystyką ImageNet.

#### Trening

- Optymalizator: **Adam**, `lr = 1e-3`.
- Scheduler: `ReduceLROnPlateau` (`factor=0.5`, `patience=3`).
- Loss: **BCE-with-logits + Dice** (oba maskowane przez FOV).
- Liczba epok: **30**. Zapisany checkpoint: najlepszy po G-mean na inner-val.

#### Predykcja na pełnym obrazie

- Tiling 256×256 z overlapem 32 pikseli; uśrednianie prawdopodobieństw na nakładkach; binaryzacja progiem 0.5.

#### Wyniki hold-out (z notebooka `03_deep_learning.ipynb`)

> _Wpisać końcowe wartości train/val loss i val G-mean z ostatniej epoki:_

| epoka      | train_loss   | val_loss     | val_gmean    |
| ---------- | ------------ | ------------ | ------------ |
| _ostatnia_ | \_0.\_\_\_\_ | \_0.\_\_\_\_ | \_0.\_\_\_\_ |

#### Uzasadnienie

U-Net jest standardem w segmentacji medycznej - skip-connections pozwalają łączyć kontekst globalny (głębokie warstwy) z precyzją pikselową (płytkie warstwy), co jest kluczowe dla cienkich struktur jak naczynia. Połączony loss BCE+Dice radzi sobie z dysbalansem klas (Dice) i jednocześnie wymusza ostrość pikselową (BCE).

---

## 4. Wizualizacja wyników działania programu

Wizualizacje (oryginał, maska ekspercka, predykcja, overlay) dla wszystkich 6 obrazów testowych - zrzuty ekranu z notebooków `01_baseline.ipynb` (sekcja 3.2), `02_classical_ml.ipynb` (sekcja 6.3), `03_deep_learning.ipynb` (sekcja 5.2). W `app.ipynb` można je obejrzeć łącznie dla wszystkich trzech metod jednocześnie.

### 4.1. Obraz `14_h` (zdrowy)

> _Wstawić obraz / zrzut: original | ground truth | baseline | RF | U-Net._

### 4.2. Obraz `15_h` (zdrowy)

> _Wstawić obraz / zrzut._

### 4.3. Obraz `14_dr` (retinopatia cukrzycowa)

> _Wstawić obraz / zrzut._

### 4.4. Obraz `15_dr` (retinopatia cukrzycowa)

> _Wstawić obraz / zrzut._

### 4.5. Obraz `14_g` (jaskra)

> _Wstawić obraz / zrzut._

### 4.6. Obraz `15_g` (jaskra)

> _Wstawić obraz / zrzut._

Wszystkie 6 obrazów testowych nie były używane do uczenia w etapie 2 ani w etapie 3 - wymóg DnoOka.md (obrazy testowe ≠ obrazy uczące) jest spełniony.

---

## 5. Analiza wyników

### 5.1. Tabela zbiorcza (mean ± std po obrazach testowych)

> _Tabelę przeklejać z `app.ipynb`, sekcja 3.1._

| metoda            | accuracy        | sensitivity     | specificity     | precision       | g_mean          | balanced_acc    |
| ----------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| baseline (Frangi) | \_0.\_**_±0._** | \_0.\_**_±0._** | \_0.\_**_±0._** | \_0.\_**_±0._** | \_0.\_**_±0._** | \_0.\_**_±0._** |
| random_forest     | \_0.\_**_±0._** | \_0.\_**_±0._** | \_0.\_**_±0._** | \_0.\_**_±0._** | \_0.\_**_±0._** | \_0.\_**_±0._** |
| unet              | \_0.\_**_±0._** | \_0.\_**_±0._** | \_0.\_**_±0._** | \_0.\_**_±0._** | \_0.\_**_±0._** | \_0.\_**_±0._** |

### 5.2. Wyniki per-image × per-method

> _Tabelę przeklejać z `app.ipynb`, sekcja 3.2._

| obraz | metoda        | accuracy     | sensitivity  | specificity  | g_mean       |
| ----- | ------------- | ------------ | ------------ | ------------ | ------------ |
| 14_h  | baseline      | \_0.\_\_\_\_ | \_0.\_\_\_\_ | \_0.\_\_\_\_ | \_0.\_\_\_\_ |
| 14_h  | random_forest | \_0.\_\_\_\_ | \_0.\_\_\_\_ | \_0.\_\_\_\_ | \_0.\_\_\_\_ |
| 14_h  | unet          | \_0.\_\_\_\_ | \_0.\_\_\_\_ | \_0.\_\_\_\_ | \_0.\_\_\_\_ |
| ...   | ...           | ...          | ...          | ...          | ...          |

### 5.3. Analiza indywidualna obrazów

#### `14_h`

> _Krótki komentarz: na którym obrazie który metod sobie poradził najlepiej / najgorzej. Co wpłynęło na wynik (jasność, patologie, drobne naczynia)._

#### `15_h`

> _Komentarz._

#### `14_dr`

> _Komentarz._

#### `15_dr`

> _Komentarz._

#### `14_g`

> _Komentarz._

#### `15_g`

> _Komentarz._

### 5.4. Porównanie metod

> _Krótkie podsumowanie różnic:_
>
> - _Czy U-Net wygrał we wszystkich metrykach?_
> - _Gdzie Random Forest jest porównywalny z U-Netem?_
> - _Na ile baseline odbiega od metod uczących się - i czy na obrazach z patologiami różnica rośnie?_
> - _Jak wygląda kompromis sensitivity vs specificity między metodami?_

---

## 6. Wnioski

> _2–3 zdania podsumowujące - która metoda okazała się najlepsza ogólnie, jaki jest jej koszt obliczeniowy względem baseline, jakie są jej ograniczenia._
