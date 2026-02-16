# 📄 Documentazione del Progetto — SVD + YOLO Image Steganography

> **Progetto**: Metodi Statistici — Magistrale  
> **Obiettivo**: Implementare un sistema di steganografia semantica basato su SVD "from scratch" e YOLOv8 per la selezione della ROI.  
> **Ultimo aggiornamento**: 16 Febbraio 2026

---

## Indice

1. [Panoramica del Progetto](#1-panoramica-del-progetto)
2. [Architettura e Struttura del Codice](#2-architettura-e-struttura-del-codice)
3. [Fase 1 — SVD from Scratch](#3-fase-1--svd-from-scratch)
4. [Fase 2 — Selezione ROI con YOLOv8](#4-fase-2--selezione-roi-con-yolov8)
5. [Fase 3 — Embedding del Messaggio](#5-fase-3--embedding-del-messaggio)
6. [Fase 4 — Estrazione del Messaggio](#6-fase-4--estrazione-del-messaggio)
7. [Pipeline Completa (Flusso Dati)](#7-pipeline-completa-flusso-dati)
8. [Dipendenze e Ambiente](#8-dipendenze-e-ambiente)
9. [Stato Attuale e Fasi Rimanenti](#9-stato-attuale-e-fasi-rimanenti)

---

## 1. Panoramica del Progetto

Il progetto implementa un **sistema di steganografia semantica** che permette di nascondere un messaggio all'interno di un'immagine in modo impercettibile all'occhio umano. Si articola in due pilastri fondamentali:

| Componente | Ruolo |
|------------|-------|
| **SVD custom** | Motore matematico per la decomposizione e manipolazione dei valori singolari (embedding/estrazione) |
| **YOLOv8** | Modello di Deep Learning per l'individuazione automatica della ROI (Region of Interest) |

Il messaggio viene nascosto modificando i **valori singolari** dei blocchi della ROI tramite **Quantization Index Modulation (QIM)**, una tecnica che consente l'estrazione blind (senza bisogno dell'immagine originale).

---

## 2. Architettura e Struttura del Codice

```
Progetto-Statistical-Method/
├── main.py                  # Entry point CLI — orchestra tutte le fasi
├── src/
│   ├── __init__.py
│   ├── svd.py              # [Fase 1] SVD implementata da zero
│   ├── image_utils.py      # Utility per immagini (caricamento, blocchi, padding)
│   ├── validation.py       # Validazione SVD custom vs numpy.linalg.svd
│   ├── yolo_roi.py         # [Fase 2] Integrazione YOLOv8 + selezione ROI
│   └── steganography.py    # [Fase 3/4] Embedding e estrazione QIM
├── output/                  # Output generato (immagini, ricostruzioni)
├── yolov8n.pt              # Modello YOLOv8 nano pre-addestrato
├── requirements.txt         # Dipendenze Python
├── README.md               # Roadmap del progetto
└── USAGE.md                # Guida all'esecuzione dettagliata
```

### Diagramma dei Moduli

```
 main.py (CLI)
   │
   ├── src/svd.py            ← Algoritmo SVD
   ├── src/image_utils.py    ← I/O immagini + blocchi
   ├── src/validation.py     ← Test correttezza
   ├── src/yolo_roi.py       ← YOLO + ROI
   └── src/steganography.py  ← Embedding/Estrazione QIM
```

---

## 3. Fase 1 — SVD from Scratch

**Modulo**: `src/svd.py` (345 righe) | **Commit**: `3192326`

### Obiettivo
Implementare la Singular Value Decomposition **senza usare `numpy.linalg.svd`**, partendo dai fondamenti matematici.

### Algoritmo Implementato

La SVD fattorizza una matrice **X** (m×n) come:

```
X = U · Σ · Vᵀ
```

Dove:
- **U** (m×m): vettori singolari sinistri (colonne ortonormali)
- **Σ** (m×n): matrice diagonale con i valori singolari in ordine decrescente
- **Vᵀ** (n×n): vettori singolari destri trasposti

### Fasi dell'Algoritmo

| Step | Operazione | Dettagli |
|------|-----------|----------|
| 1 | **Matrice di covarianza** | C = XᵀX |
| 2 | **Autovalori/Autovettori di C** | Tramite *metodo delle potenze* con deflazione → matrice **V** |
| 3 | **Matrice di Gram** | S = XXᵀ → autodecomposizione per trovare **U** |
| 4 | **Valori singolari** | σᵢ = √(λᵢ) dalle radici degli autovalori |
| 5 | **Validazione** | Confronto con `numpy.linalg.svd` |

### Funzioni Principali

| Funzione | Descrizione |
|----------|-------------|
| `_power_method()` | Metodo delle potenze per trovare l'autovalore dominante. Include ri-ortogonalizzazione Gram-Schmidt |
| `_orthogonalize()` | Ortogonalizzazione Gram-Schmidt a due passi per stabilità numerica |
| `_eigen_decomposition()` | Calcola i primi *k* autovalori/autovettori con deflazione |
| `svd()` | SVD completa (m×m U, tutti i σ, n×n Vᵀ) |
| `svd_compact()` | SVD in forma economy/thin (solo componenti non nulle) |
| `reconstruct()` | Ricostruzione X ≈ U[:,:k] · Σ[:k] · Vᵀ[:k,:] con approssimazione di rango k |

### Validazione (`src/validation.py`)

Il modulo di validazione testa la SVD custom su **6 tipi di matrici**:
1. Matrice casuale (5×3)
2. Matrice casuale (3×5) — verifica caso m < n
3. Matrice quadrata (4×4)
4. Matrice di rango 1
5. Matrice di rango 2
6. Matrice grande (50×30)

**Metriche verificate**:
- Errore ricostruzione: ‖X − UΣVᵀ‖ ≈ 0
- Ortonormalità di U: UᵀU ≈ I
- Ortonormalità di V: VᵀV ≈ I
- Errore relativo sui valori singolari rispetto a NumPy

### Utility Immagini (`src/image_utils.py`)

| Funzione | Descrizione |
|----------|-------------|
| `load_image_as_matrix()` | Carica immagine come matrice numpy float64 (grayscale o RGB) |
| `matrix_to_image()` | Converte matrice → PIL Image (clip [0,255], uint8) |
| `save_image()` | Salva matrice come file immagine |
| `apply_mean_centering()` | Sottrae la media per colonna (feature), ritorna dati + medie |
| `remove_mean_centering()` | Riaggiunge le medie per ripristinare i valori originali |
| `split_into_blocks()` | Suddivide matrice in blocchi N×N con zero-padding |
| `merge_blocks()` | Ricompone la matrice dai blocchi rimuovendo il padding |

---

## 4. Fase 2 — Selezione ROI con YOLOv8

**Modulo**: `src/yolo_roi.py` (394 righe) | **Commit**: `04d18b3`

### Obiettivo
Integrare YOLOv8 per individuare automaticamente gli oggetti nell'immagine e selezionare la regione dove nascondere il messaggio.

### Data Types

```python
class BoundingBox:
    """Rappresenta un bounding box rilevato da YOLO."""
    x1, y1, x2, y2: int       # Coordinate rettangolo
    confidence: float          # Confidenza della detection
    class_id: int              # ID della classe COCO
    class_name: str            # Nome leggibile (es. "person", "car")
    # Properties: width, height, area

class ROIResult:
    """Risultato della selezione ROI."""
    mask: np.ndarray           # Maschera booleana (H×W), True = pixel ROI
    bounding_boxes: list       # Tutti i box rilevati
    strategy: str              # Strategia usata (A/B/C)
    selected_box: BoundingBox  # Box selezionato (None per strategia B)
```

### Tre Strategie di Selezione

| Strategia | Descrizione | Caso d'uso |
|-----------|-------------|------------|
| **A — Soggetti** | Nasconde nei bounding box (la texture complessa maschera le alterazioni) | Immagini con soggetti dettagliati |
| **B — Sfondo** | Nasconde nello sfondo, escludendo i bounding box | Preservare i soggetti principali |
| **C — Automatico** | Sceglie il bounding box più grande (default) | Massimizzare la capacità automaticamente |

### Funzioni Principali

| Funzione | Descrizione |
|----------|-------------|
| `load_yolo_model()` | Carica il modello YOLOv8 pre-addestrato |
| `detect_objects()` | Esegue inferenza YOLO, restituisce lista di `BoundingBox` ordinata per area |
| `select_roi()` | Seleziona la ROI in base alla strategia A/B/C e restituisce un `ROIResult` |
| `extract_roi_region()` | Estrae la porzione rettangolare dell'immagine corrispondente alla ROI |
| `draw_detections()` | Disegna i bounding box sull'immagine con annotazioni |
| `print_detection_report()` | Stampa un report leggibile delle detection |

### Parametri Configurabili

| Parametro | Default | Effetto |
|-----------|---------|--------|
| `--yolo-model` | `yolov8n.pt` | Modello da usare (n=nano, s=small, m=medium) |
| `--yolo-confidence` | `0.25` | Soglia minima di confidenza per le detection |
| `--strategy` | `C` | Strategia A, B o C |
| `--box-index` | `None` | Indice del box specifico (solo strategia A) |

---

## 5. Fase 3 — Embedding del Messaggio

**Modulo**: `src/steganography.py` (623 righe) | **Commit**: `3056f81`

### Obiettivo
Fondere i dati del messaggio all'interno dei valori singolari della ROI in modo impercettibile.

### Algoritmo: Quantization Index Modulation (QIM)

Il QIM è una tecnica di watermarking/steganografia che modifica i valori singolari in modo **deterministico e reversibile**:

```
Embedding di un bit b ∈ {0, 1} nel valore singolare σ:

1. Calcola il quoziente q = ⌊σ / Δ⌋
2. Se q mod 2 ≠ b → sposta σ al multiplo di Δ che soddisfa la parità desiderata
3. σ_modificato = (q_nuovo + 0.5) · Δ   (centra nel bin di quantizzazione)
```

**Estrazione** (blind):
```
1. Calcola q = ⌊σ / Δ⌋
2. bit = q mod 2
```

### Pipeline di Embedding

```
Messaggio "Hello"
      │
      ▼
┌──────────────────────┐
│  text_to_binary()    │  Converte in sequenza binaria + terminatore 0x00
│  "Hello" → [01001000, │  01100101, ...]
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ split_into_blocks()  │  ROI → blocchi 8×8 (con zero-padding)
└──────────┬───────────┘
           │
           ▼        Per ogni blocco:
┌──────────────────────┐
│ svd_compact(blocco)  │  → U, Σ, Vᵀ (SVD custom)
│ _get_sv_indices()    │  Seleziona quali SV modificare (first/mid/last)
│ _qim_embed(σᵢ, bit)  │  Incorpora 1 bit per SV selezionato
│ reconstruct(U,Σ',Vᵀ) │  → blocco modificato
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ merge_blocks()       │  Ricompone la ROI modificata
│ embed_in_full_image()│  Inserisce ROI nella cover image → stego-image
└──────────────────────┘
```

### Conversione Messaggio ↔ Binario

| Funzione | Descrizione |
|----------|-------------|
| `text_to_binary()` | Testo → array di bit (8 bit/carattere + terminatore 0x00) |
| `binary_to_text()` | Array di bit → testo (si ferma al terminatore) |
| `image_to_binary()` | Immagine → bit (header 32 bit con dimensioni + pixel) |
| `binary_to_image()` | Bit → matrice immagine |

### Calcolo Capacità (`compute_capacity()`)

La capacità di embedding dipende da:
- **Dimensione ROI** (pixel disponibili)
- **Block size** (8×8 → 64 pixel/blocco)
- **SV range** (first/mid/last → ⅓ dei SV per blocco)

```
n_blocchi = (W_roi / block_size) × (H_roi / block_size)
bits_per_blocco = |SV utilizzabili nel range scelto|
capacità_totale = n_blocchi × bits_per_blocco
max_caratteri = capacità_totale / 8 - 1  (terminatore)
```

### Parametri di Embedding

| Parametro | Default | Effetto |
|-----------|---------|--------|
| `--block-size` | `8` | Dimensione dei blocchi (4, 8 o 16) |
| `--sv-range` | `mid` | Quali SV alterare: `first` (robusti), `mid` (compromesso), `last` (invisibili) |
| `--delta` | `15.0` | Passo di quantizzazione QIM: più alto = più robusto ma più visibile |

### Trade-off SV Range

```
            Robustezza ←────────────────→ Invisibilità
            ┌──────────┬──────────┬──────────┐
SV Range:   │  first   │   mid    │   last   │
            │ Artefatti│Compromesso│ Invisibile│
            │ visibili │  ottimo  │ Fragile  │
            └──────────┴──────────┴──────────┘
```

---

## 6. Fase 4 — Estrazione del Messaggio

**Modulo**: `src/steganography.py` | **Commit**: `b8dadd8`

### Obiettivo
Recuperare il messaggio nascosto dalla stego-image **senza bisogno dell'immagine originale** (estrazione blind).

### Pipeline di Estrazione

```
Stego-Image
      │
      ▼
┌──────────────────────┐
│ YOLO: detect_objects()│  Rileva la ROI (stesse coordinate dell'embedding)
│ select_roi()          │
└──────────┬────────────┘
           │
           ▼
┌──────────────────────┐
│ split_into_blocks()  │  ROI stego → blocchi 8×8
└──────────┬───────────┘
           │
           ▼        Per ogni blocco:
┌──────────────────────┐
│ svd_compact(blocco)  │  → U', Σ', Vᵀ'
│ _qim_extract(σᵢ, Δ)  │  Estrae bit dalla parità del quoziente
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ binary_to_text()     │  Array bit → testo (fino al terminatore 0x00)
└──────────────────────┘
```

### Vincoli Fondamentali

> ⚠️ **I parametri di estrazione devono corrispondere esattamente a quelli di embedding**:
> `--block-size`, `--sv-range`, `--delta`, `--strategy`

---

## 7. Pipeline Completa (Flusso Dati)

```
                    EMBEDDING                                ESTRAZIONE
                    ────────                                 ──────────

   Cover Image ──────────────────┐                    Stego Image ──────────┐
        │                        │                         │                │
        ▼                        │                         ▼                │
   ┌─────────┐                   │                    ┌─────────┐           │
   │  YOLO   │ detect_objects()  │                    │  YOLO   │           │
   │ YOLOv8n │                   │                    │ YOLOv8n │           │
   └────┬────┘                   │                    └────┬────┘           │
        │ Bounding Boxes         │                         │ Bounding Boxes │
        ▼                        │                         ▼                │
   ┌─────────┐                   │                    ┌─────────┐           │
   │Select   │ Strategia A/B/C  │                    │Select   │           │
   │  ROI    │                   │                    │  ROI    │           │
   └────┬────┘                   │                    └────┬────┘           │
        │ ROI matrix             │                         │ Stego ROI      │
        ▼                        │                         ▼                │
   ┌─────────┐ Blocchi 8×8       │                    ┌─────────┐           │
   │  Split  │──────────┐        │                    │  Split  │──────┐    │
   └─────────┘          │        │                    └─────────┘      │    │
                        ▼        │                                     ▼    │
   Messaggio ──→ ┌──────────┐    │                              ┌──────────┐│
                 │SVD custom│    │                              │SVD custom││
   "Hello" ───→ │ + QIM    │    │                              │ + QIM    ││
                 │ embed   │    │                              │ extract  ││
                 └────┬─────┘    │                              └────┬─────┘│
                      │          │                                   │      │
                      ▼          │                                   ▼      │
                 ┌──────────┐    │                              ┌──────────┐│
                 │  Merge   │    │                              │ Decode   ││
                 │ Blocks   │    │                              │  Bits    ││
                 └────┬─────┘    │                              └────┬─────┘│
                      │          │                                   │
                      ▼          │                                   ▼
                 Stego Image ◄───┘                              "Hello"
```

---

## 8. Dipendenze e Ambiente

### Requisiti di Sistema
- **Python** ≥ 3.10
- **pip** (gestore pacchetti)
- Virtual environment (`.venv`)

### Librerie (`requirements.txt`)

| Pacchetto | Versione | Utilizzo |
|-----------|----------|----------|
| `numpy` | ≥ 1.24.0 | Algebra lineare, manipolazione matrici |
| `Pillow` | ≥ 10.0.0 | Caricamento/salvataggio immagini |
| `matplotlib` | ≥ 3.7.0 | Visualizzazione (ricostruzioni SVD) |
| `ultralytics` | ≥ 8.0.0 | YOLOv8 object detection |

### Modello YOLO
- **File**: `yolov8n.pt` (6.5 MB) — modello "nano", il più leggero
- **Dataset**: Pre-addestrato su COCO (80 classi di oggetti)
- **Alternative**: yolov8s.pt (small), yolov8m.pt (medium) per maggiore accuratezza

---

## 9. Stato Attuale e Fasi Rimanenti

### ✅ Fasi Completate

| Fase | Stato | Commit | Descrizione |
|------|-------|--------|-------------|
| **Fase 1** — SVD from Scratch | ✅ Completata | `3192326` | Implementazione completa con validazione |
| **Fase 2** — YOLO ROI Selection | ✅ Completata | `04d18b3` | 3 strategie (A/B/C), parametri avanzati |
| **Fase 3** — Embedding | ✅ Completata | `3056f81` | QIM con scelta SV range, blocchi, delta |
| **Fase 4** — Estrazione | ✅ Completata | `b8dadd8` | Estrazione blind funzionante |

### ⬜ Fase Rimanente

| Fase | Stato | Descrizione |
|------|-------|-------------|
| **Fase 5** — Valutazione Performance | ⬜ Da fare | Metriche visive (PSNR, SSIM) e test di robustezza (filtri, rumore, JPEG) |

### Dettagli Fase 5 (Da Implementare)

1. **Metriche Visive**:
   - **PSNR** (Peak Signal-to-Noise Ratio): misura la distorsione tra cover e stego-image
   - **SSIM** (Structural Similarity Index): misura la similarità strutturale percepita
   - Nota: il PSNR è già calcolato in `validation.py` (`_compute_psnr()`) e nel report di embedding

2. **Metriche di Robustezza**:
   - Applicare filtri (blur, sharpening) alla stego-image e verificare l'estrazione
   - Aggiungere rumore gaussiano e testare la decodifica
   - Comprimere in JPEG a diversi livelli di qualità e verificare:
     - Se il payload è ancora estraibile
     - Se YOLO riconosce ancora i bounding box

---

## Riepilogo Quantitativo

| Metrica | Valore |
|---------|--------|
| **Moduli Python** | 5 (`svd`, `image_utils`, `validation`, `yolo_roi`, `steganography`) |
| **Entry point** | 1 (`main.py`, 761 righe) |
| **Righe di codice totali** | ~2.400 (src/ + main.py) |
| **Argomenti CLI** | 14 flag configurabili |
| **Commit** | 6 (dal setup iniziale alla Fase 4) |
| **Fasi completate** | 4/5 |
