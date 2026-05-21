# Wykrywanie naczyń dna siatkówki oka - raport

**Przedmiot:** Informatyka w Medycynie, projekt DnoOka.

## 1. Skład grupy

- Illia Yanukovich 159788
- Vladyslav Vatslavyi 161317

---

## 2. Zastosowany język programowania oraz dodatkowe biblioteki

Projekt napisany w Pythonie 3.10+, w formie czterech notebooków Jupyter w katalogu `notebooks/`:

- `01_baseline.ipynb` - przetwarzanie obrazu (Frangi + morfologia), realizacja wymagań obowiązkowych.
- `02_classical_ml.ipynb` - Random Forest na patchach 5×5, wymagania na 4.0.
- `03_deep_learning.ipynb` - U-Net w PyTorch, wymagania na 5.0.
- `app.ipynb` - interaktywna aplikacja i porównanie trzech metod.

Wspólny kod znajduje się w modułach `src/*.py`.

Wykorzystane biblioteki (pełna lista w `requirements.txt`):

- NumPy, SciPy do operacji numerycznych.
- scikit-image (filtr Frangiego, próg Otsu, operacje morfologiczne).
- OpenCV (CLAHE, momenty Hu, gradient Sobela).
- Pillow do wczytywania obrazów i masek.
- scikit-learn (Random Forest, podział train/val, macierz pomyłek).
- imbalanced-learn (`RandomUnderSampler` do zrównoważenia klas).
- PyTorch wraz z `segmentation-models-pytorch`, `albumentations`, `MONAI` i `PyTorch Lightning` do implementacji, treningu i inferencji U-Net.
- pandas, matplotlib, ipywidgets, tqdm do wizualizacji, tabel, widgetów i paska postępu.

**Baza obrazów:** HRF Image Database (<https://www5.cs.fau.de/research/data/fundus-images/>), 45 obrazów dna oka w trzech kategoriach: 15 zdrowych (`h`), 15 z retinopatią cukrzycową (`dr`), 15 z jaskrą (`g`). Każdy obraz posiada ekspercką maskę naczyń (`manual1/`) i maskę pola widzenia FOV (`mask/`). Ta sama baza używana jest we wszystkich trzech etapach.

**Podział hold-out** (ustalony w `src/config.py`, identyczny dla wszystkich metod):

- **Train:** 39 obrazów.
- **Test:** 6 obrazów: `14_h`, `15_h`, `14_dr`, `15_dr`, `14_g`, `15_g` (po dwa z każdej kategorii).

---

## 3. Opis zastosowanych metod

### 3.1. Przetwarzanie obrazów (wymagania obowiązkowe)

#### Kroki przetwarzania

1. Wstępne przetwarzanie (`src/preprocessing.py`):
   - Wyodrębnienie kanału zielonego, na którym naczynia mają największy kontrast.
   - Wypełnienie pikseli poza FOV medianą pikseli wewnątrz FOV. Bez tego twarda krawędź ramki byłaby wykrywana przez filtr Frangiego jako fałszywe naczynie.
   - CLAHE (`cv2.createCLAHE`, `clipLimit=2.0`, `tileGridSize=(8,8)`) do lokalnego wyrównania kontrastu z ograniczeniem wzmocnienia szumu.

2. Właściwe przetwarzanie (`src/baseline.py`):
   - Filtr Frangiego (`skimage.filters.frangi`, `sigmas=(1,2,3,4,5)`, `black_ridges=True`) jako detektor struktur tubularnych w wielu skalach.
   - Normalizacja odpowiedzi do `[0, 1]`.
   - Próg Otsu (`skimage.filters.threshold_otsu`) liczony wyłącznie po pikselach FOV. Uwzględnienie czarnego tła ramki przesunęłoby próg w stronę zera i naczynia zostałyby utracone.

3. Końcowe przetwarzanie:
   - Usunięcie małych komponentów spójności (`skimage.morphology.remove_small_objects`, `min_size=60`) w celu wyeliminowania izolowanych pikseli szumu.
   - Zamknięcie morfologiczne dyskiem o promieniu 1, żeby połączyć drobne przerwy w naczyniach.
   - Końcowe obcięcie maski do FOV.

#### Uzasadnienie

Filtr Frangiego to klasyczny detektor naczyń o dowolnej orientacji, oparty na analizie wartości własnych hesjanu obrazu w wielu skalach. W parze z CLAHE i progowaniem Otsu (bez parametrów do ręcznego dobrania) daje sensowny baseline bez uczenia maszynowego. Operacje morfologiczne dodatkowo czyszczą wynik z drobnego szumu.

### 3.2. Klasyczne uczenie maszynowe (wymagania na 4.0)

#### Przygotowanie danych - wycinki i ekstrakcja cech

Z każdego obrazu uczącego losujemy 5000 patchy 5×5 wewnątrz FOV. Etykietą patcha jest wartość maski eksperckiej w jego środkowym pikselu. Z każdego patcha wyciągamy wektor 24 cech:

- 6 statystyk RGB: średnia i wariancja w każdym z trzech kanałów.
- 2 cechy środkowego piksela: kanał zielony surowy i po CLAHE.
- 4 statystyki patcha na kanale zielonym: mean, var, min, max.
- 7 momentów Hu (`cv2.HuMoments`) z patcha binaryzowanego progiem średniej jasności, niezmiennicze względem skali i obrotu.
- 3 momenty centralne (`mu20`, `mu02`, `mu11`).
- 2 statystyki magnitudy gradientu Sobela: mean i var.

#### Wstępne przetwarzanie zbioru uczącego

Pełny zbiór jest dzielony stratyfikowanie 2/3 : 1/3 na train/val. Na części uczącej stosujemy undersampling klasy „tło" (`imblearn.under_sampling.RandomUnderSampler`, `sampling_strategy='auto'`), żeby zrównać liczność klas do 50/50. Resampling działa wyłącznie podczas `fit`, dzięki opakowaniu w `imblearn.Pipeline(undersample -> rf)`.

#### Klasyfikator i parametry

Random Forest (`sklearn.ensemble.RandomForestClassifier`):

- `n_estimators = 100`
- `max_depth = None`
- `min_samples_leaf = 2`
- `random_state = 42`

#### Wyniki hold-out (z notebooka `02_classical_ml.ipynb`)

Średnie po 6 obrazach testowych:

| accuracy        | sensitivity     | specificity     | g_mean          |
| --------------- | --------------- | --------------- | --------------- |
| 0.8755 ± 0.0383 | 0.7659 ± 0.0773 | 0.8865 ± 0.0493 | 0.8219 ± 0.0212 |

#### Uzasadnienie

Random Forest został wybrany jako klasyfikator odporny na nieliniowe zależności między cechami i niewymagający skalowania danych wejściowych (w przeciwieństwie do np. SVM). Wybrane cechy łączą informację o kolorze (statystyki RGB i G), kształcie (momenty Hu i centralne) oraz krawędziach (gradient Sobela), więc klasyfikator dysponuje kilkoma komplementarnymi reprezentacjami patcha.

### 3.3. Głębokie uczenie (wymagania na 5.0)

#### Architektura

U-Net z biblioteki `segmentation_models_pytorch`, encoder ResNet-34 pretrenowany na ImageNet, dekoder inicjalizowany losowo. Wejście: 3 kanały RGB. Wyjście: 1 kanał (logit prawdopodobieństwa naczynia). Łącznie 24.4 mln parametrów.

#### Przygotowanie danych

- Wewnątrz puli 39 obrazów uczących wydzielony inner-val (6 obrazów) do monitorowania G-mean w trakcie treningu i wyboru najlepszego checkpointa. Zbiór `TEST_IMAGES` pozostaje nietknięty do końcowej oceny.
- Z każdego obrazu losowe kropy 256×256 wewnątrz FOV, po 64 kropy na obraz na epokę.
- Sampling środków kropów z biasem na piksele naczyniowe (`vessel_prob = 0.5`). Dzięki temu połowa kropów uczących zawiera naczynia, mimo że stanowią one tylko ~10 % pikseli obrazu.
- Augmentacje (`albumentations`): obroty o `k·90°`, odbicie poziome i pionowe.
- Normalizacja per-kanał statystyką ImageNet, wymagana przez pretrenowany encoder.

#### Trening

- Optymalizator Adam z `lr = 1e-3`.
- Scheduler `ReduceLROnPlateau` (`factor=0.5`, `patience=3`).
- Loss: BCE-with-logits + Dice, oba maskowane przez FOV.
- Liczba epok: 30, batch size 8.
- Pętla treningowa zaimplementowana przez PyTorch Lightning.
- Zapisywany checkpoint to najlepszy po G-mean na inner-val (osiągnięty w epoce 4, val_gmean = 0.9021).

#### Predykcja na pełnym obrazie

Sliding-window inference z `monai.inferers.sliding_window_inference`: tiling 256×256 z overlapem 32 px, uśrednianie prawdopodobieństw na nakładkach, binaryzacja progiem 0.5, końcowe obcięcie do FOV.

#### Uzasadnienie

U-Net jest standardem w segmentacji medycznej. Skip-connections pozwalają łączyć kontekst globalny (głębokie warstwy encoder'a) z precyzją pikselową (płytkie warstwy), co jest kluczowe dla cienkich struktur takich jak naczynia. Pretrenowany encoder ImageNet jest praktycznie konieczny, ponieważ 33 obrazy uczące to za mały zbiór, by uczyć U-Net od zera. Połączony loss BCE+Dice radzi sobie z dysbalansem klas (Dice) i jednocześnie wymusza dokładność pikselową (BCE).

---

## 4. Wizualizacja wyników

Dla każdego z 6 obrazów testowych poniżej zaprezentowano oryginał, maskę ekspercką oraz predykcje trzech metod (baseline / Random Forest / U-Net). Wszystkie obrazy testowe są rozłączne ze zbiorem uczącym we wszystkich etapach.

### 4.1. Obraz `14_h` (zdrowy)

![14_h](figs/14_h.png)

### 4.2. Obraz `15_h` (zdrowy)

![15_h](figs/15_h.png)

### 4.3. Obraz `14_dr` (retinopatia cukrzycowa)

![14_dr](figs/14_dr.png)

### 4.4. Obraz `15_dr` (retinopatia cukrzycowa)

![15_dr](figs/15_dr.png)

### 4.5. Obraz `14_g` (jaskra)

![14_g](figs/14_g.png)

### 4.6. Obraz `15_g` (jaskra)

![15_g](figs/15_g.png)

---

## 5. Analiza wyników

### 5.1. Wyniki zbiorcze (mean ± std po 6 obrazach testowych)

| metoda            | accuracy        | sensitivity     | specificity     | g_mean          |
| ----------------- | --------------- | --------------- | --------------- | --------------- |
| baseline (Frangi) | 0.9094 ± 0.0245 | 0.6645 ± 0.0462 | 0.9332 ± 0.0273 | 0.7870 ± 0.0295 |
| random_forest     | 0.8755 ± 0.0383 | 0.7659 ± 0.0773 | 0.8865 ± 0.0493 | 0.8219 ± 0.0212 |
| unet              | 0.9629 ± 0.0078 | 0.8165 ± 0.0256 | 0.9772 ± 0.0086 | 0.8932 ± 0.0154 |

![Porównanie metod](comparison_bars.png)

### 5.2. Wyniki indywidualne (per obraz × per metoda)

| obraz | metoda        | accuracy | sensitivity | specificity | g_mean |
| ----- | ------------- | -------- | ----------- | ----------- | ------ |
| 14_h  | baseline      | 0.9018   | 0.7381      | 0.9206      | 0.8243 |
| 14_h  | random_forest | 0.8870   | 0.7809      | 0.8992      | 0.8380 |
| 14_h  | unet          | 0.9692   | 0.8488      | 0.9830      | 0.9134 |
| 15_h  | baseline      | 0.9363   | 0.6598      | 0.9645      | 0.7977 |
| 15_h  | random_forest | 0.9257   | 0.6668      | 0.9522      | 0.7968 |
| 15_h  | unet          | 0.9760   | 0.8229      | 0.9916      | 0.9033 |
| 14_dr | baseline      | 0.8953   | 0.5964      | 0.9250      | 0.7428 |
| 14_dr | random_forest | 0.8533   | 0.7813      | 0.8604      | 0.8199 |
| 14_dr | unet          | 0.9577   | 0.7851      | 0.9748      | 0.8749 |
| 15_dr | baseline      | 0.8711   | 0.6542      | 0.8893      | 0.7627 |
| 15_dr | random_forest | 0.8349   | 0.8463      | 0.8340      | 0.8401 |
| 15_dr | unet          | 0.9591   | 0.8339      | 0.9696      | 0.8992 |
| 14_g  | baseline      | 0.9229   | 0.6845      | 0.9446      | 0.8041 |
| 14_g  | random_forest | 0.9116   | 0.6793      | 0.9328      | 0.7960 |
| 14_g  | unet          | 0.9587   | 0.7869      | 0.9744      | 0.8756 |
| 15_g  | baseline      | 0.9288   | 0.6540      | 0.9549      | 0.7903 |
| 15_g  | random_forest | 0.8406   | 0.8407      | 0.8405      | 0.8406 |
| 15_g  | unet          | 0.9568   | 0.8216      | 0.9697      | 0.8926 |

### 5.3. Komentarz dla poszczególnych obrazów

Na obrazie `14_h` U-Net uzyskuje najlepszy wynik z całego zbioru testowego (g_mean 0.91, czyli o 0.07 lepiej niż RF i o 0.09 lepiej niż baseline). Obraz jest dobrze oświetlony, a naczynia kontrastowe, więc dla wszystkich trzech metod jest to scenariusz najprostszy.

Na obrazie `15_h` baseline i Random Forest dają niemal identyczne wyniki (g_mean ok. 0.80), za to U-Net wyraźnie odstaje na plus (0.90). Zdjęcie jest jaśniejsze od `14_h`, ale cienkie naczynia mają mniejszy kontrast, przez co sensitivity baseline'u i RF spada poniżej 0.67, podczas gdy U-Net wciąż wykrywa ponad 82 % naczyń.

Na obrazie `14_dr` widać największą przepaść między baseline'em a metodami uczącymi się (0.74 vs 0.82 vs 0.87). Zmiany patologiczne, charakterystyczne dla retinopatii cukrzycowej, zaburzają tło, więc filtr Frangiego gubi cienkie naczynia (sensitivity tylko 0.60). Random Forest radzi sobie lepiej, ale to dopiero U-Net dochodzi do wyniku, który da się uznać za satysfakcjonujący.

Obraz `15_dr` to ciekawy przypadek dla Random Forestu, który osiąga tu wysoką sensitivity 0.85, ale przy specificity 0.83. Oznacza to, że klasyfikator za często bierze tło za naczynia, przez co precision spada do 0.30. U-Net trzyma o wiele lepszy balans: 0.83 dla sensitivity i 0.97 dla specificity, co daje g_mean 0.90.

Na obrazie `14_g` (jaskra) jako jedynym z całego zbioru testowego baseline minimalnie wyprzedza Random Forest (g_mean 0.804 vs 0.796). Glaukoma zmienia wygląd tarczy nerwu wzrokowego i RF zaczyna w tym rejonie generować fałszywe dodatnie, podczas gdy filtr Frangiego pozostaje selektywny. U-Net jak zwykle ma najlepszy wynik (0.88).

Obraz `15_g` znowu pokazuje słabość Random Forestu na obrazach z patologią. Wysoka sensitivity 0.84 wygląda dobrze, ale specificity 0.84 oznacza, że model klasyfikuje wiele pikseli tła jako naczynia (precision tylko 0.33). U-Net trzyma 0.82 sensitivity przy 0.97 specificity, co daje g_mean 0.89.

### 5.4. Porównanie metod przetwarzania obrazu i uczenia maszynowego

Po uśrednieniu po 6 obrazach testowych U-Net uzyskuje najlepsze wartości wszystkich badanych metryk: 0.9629 accuracy, 0.8165 sensitivity, 0.9772 specificity i 0.8932 g_mean. Ma również najmniejsze odchylenie standardowe między obrazami (0.015 dla g_mean, RF 0.021, baseline 0.030), więc utrzymuje stabilny poziom niezależnie od kategorii obrazu (zdrowy, retinopatia, jaskra).

Random Forest plasuje się pośrodku, ale jego charakterystyka różni się od baseline'u: ma znacznie wyższą sensitivity (0.77 vs 0.66) kosztem niższej specificity (0.89 vs 0.93). RF znajduje więcej naczyń, ale jednocześnie generuje sporo fałszywych dodatnich. Widać to wyraźnie na obrazach z patologiami (`15_dr`, `15_g`), gdzie sensitivity rośnie do 0.85, ale specificity spada do 0.83-0.84.

Baseline (Frangi + Otsu) zachowuje wysoką specificity (0.93), ponieważ filtr Frangiego z założenia jest selektywny i nie wykrywa naczyń tam, gdzie ich nie ma. Płaci za to niską sensitivity (0.66), bo gubi naczynia cienkie i słabo kontrastowe. To metoda dobra do szybkiego, zachowawczego baseline'u, ale samodzielnie nie nadaje się tam, gdzie zależy nam na czułości.

Wartości g_mean (baseline 0.79, RF 0.82, U-Net 0.89) układają się w kolejności zgodnej z rosnącą złożonością modeli: prosty filtr obrazu, model uczący się z ręcznie zaprojektowanymi cechami, sieć ucząca się reprezentacji z surowych pikseli. Razem z jakością rośnie też koszt obliczeniowy: baseline daje wynik w ~1 sekundę na obraz, Random Forest potrzebuje ~30 s przy `stride=2`, a U-Net wymagał ~9 godzin treningu na układzie Apple M (sama predykcja na obrazie testowym to już tylko ~3 s).
