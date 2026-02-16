# Guida all'Esecuzione

Questo documento fornisce le istruzioni per configurare l'ambiente ed eseguire il modulo di **SVD from scratch** (Fase 1) e il **rilevamento YOLO con selezione della ROI** (Fase 2).

## 1. Requisiti

- Python 3.10 o superiore
- pip (gestore pacchetti Python)

## 2. Configurazione Ambiente

Il progetto utilizza un ambiente virtuale (`.venv`) per gestire le dipendenze in modo isolato.

### Creazione ed Attivazione Virtual Environment

Se non è già presente, puoi crearlo con:

```bash
python3 -m venv .venv
```

**Attivazione:**

- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```
- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

### Installazione Dipendenze

Con l'ambiente attivato, installa le librerie necessarie:

```bash
pip install -r requirements.txt
```

> **Nota:** l'installazione di `ultralytics` (YOLOv8) potrebbe richiedere qualche minuto e scaricare automaticamente il modello pre-addestrato al primo utilizzo.

## 3. Fase 1 — SVD from Scratch

Il file principale è `main.py`, che funge da interfaccia CLI per tutte le funzionalità.

### Solo Validazione (Matrici Random)

Per verificare la correttezza matematica della SVD custom confrontandola con `numpy.linalg.svd`:

```bash
python main.py
```

### Demo su Immagine Reale

Per testare la scomposizione e ricostruzione di un'immagine (es. `mia_immagine.jpg`):

```bash
python main.py --image percorso/della/tua_immagine.jpg
```

I risultati della ricostruzione verranno salvati nella cartella `output/`.

### Test Mean Centering

Per analizzare l'impatto della centratura dei dati (Mean Centering) sulla qualità della ricostruzione:

```bash
python main.py --image percorso/immagine.jpg --mean-centering-test
```

## 4. Fase 2 — Rilevamento YOLO e Selezione ROI

La Fase 2 integra YOLOv8 per individuare gli oggetti nell'immagine e selezionare la Region of Interest (ROI) dove nascondere il messaggio.

### Esecuzione Base (Strategia Automatica)

```bash
python main.py --image percorso/immagine.jpg --yolo
```

Per default viene usata la **Strategia C** (sceglie automaticamente il bounding box più grande).

### Strategie di Embedding

L'utente può scegliere dove nascondere il messaggio con il flag `--strategy`:

| Strategia | Flag | Descrizione |
|-----------|------|-------------|
| **A — Soggetti** | `--strategy A` | Nasconde il messaggio dentro un bounding box (le texture complesse coprono meglio le alterazioni) |
| **B — Sfondo** | `--strategy B` | Nasconde il messaggio nello sfondo, escludendo tutti i bounding box (non altera i soggetti principali) |
| **C — Automatico** | `--strategy C` | Sceglie dinamicamente il bounding box più grande (default) |

**Esempi:**

```bash
# Strategia A — nascondere nei soggetti rilevati
python main.py --image foto.png --yolo --strategy A

# Strategia B — nascondere nello sfondo
python main.py --image foto.png --yolo --strategy B

# Strategia C — bounding box più grande (default)
python main.py --image foto.png --yolo --strategy C
```

### Selezione di un Bounding Box Specifico (Strategia A)

Se YOLO rileva più oggetti e vuoi usare uno specifico bounding box:

```bash
python main.py --image foto.png --yolo --strategy A --box-index 2
```

L'indice è basato sulla lista ordinata per area decrescente (0 = il più grande).

### Parametri Avanzati YOLO

```bash
# Usa un modello YOLO diverso (es. yolov8s per maggiore accuratezza)
python main.py --image foto.png --yolo --yolo-model yolov8s.pt

# Modifica la soglia di confidenza (default: 0.25)
python main.py --image foto.png --yolo --yolo-confidence 0.5
```

### Output Fase 2

La Fase 2 salva nella cartella `output/`:

| File | Descrizione |
|------|-------------|
| `yolo_detections.png` | Immagine originale con i bounding box disegnati |
| `roi_extracted.png` | La regione di interesse estratta |
| `roi_mask.png` | Maschera binaria della ROI (bianco = area utilizzabile) |
| `roi_reconstructed_k*.png` | Ricostruzioni SVD della ROI a diversi livelli di compressione |

## 5. Opzioni Avanzate

Visualizza tutti i parametri disponibili:

```bash
python main.py --help
```

| Opzione | Descrizione |
|---------|-------------|
| `--image`, `-i` | Percorso dell'immagine |
| `--output`, `-o` | Directory di output (default: `output`) |
| `--no-validate` | Salta i test matematici iniziali |
| `--mean-centering-test` | Testa impatto del mean centering |
| `--yolo` | Attiva la Fase 2 (YOLO + ROI) |
| `--strategy`, `-s` | Strategia di embedding: A, B, C (default: C) |
| `--box-index` | Indice del bounding box (solo strategia A) |
| `--yolo-model` | Modello YOLOv8 (default: `yolov8n.pt`) |
| `--yolo-confidence` | Soglia di confidenza YOLO (default: 0.25) |

## 6. Struttura del Progetto

- `src/svd.py`: Implementazione core dell'algoritmo SVD tramite metodo delle potenze e deflazione.
- `src/image_utils.py`: Funzioni per conversione immagini, padding, blocchi e mean centering.
- `src/validation.py`: Logica di confronto tra l'implementazione custom e quella ufficiale di NumPy.
- `src/yolo_roi.py`: **[Fase 2]** Integrazione YOLOv8 per object detection e selezione della ROI con strategie A/B/C.
- `main.py`: Entry point del programma con CLI completa.
