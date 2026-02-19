# Guida all'Esecuzione

Questo documento fornisce le istruzioni per configurare l'ambiente ed eseguire il sistema di **SVD Image Steganography**: dalla validazione SVD (Fase 1), al rilevamento YOLO con selezione ROI (Fase 2), all'embedding del messaggio (Fase 3) e all'estrazione informed (Fase 4).

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

## 3. Configurazione di `main.py`

Tutte le impostazioni del progetto sono **variabili di configurazione** definite direttamente in testa al file `main.py`. Non ci sono argomenti da riga di comando.

### Variabili Principali

Apri `main.py` e modifica la sezione `CONFIGURAZIONE`:

```python
# ═══ CONFIGURAZIONE ═══

# Percorso dell'immagine di input (cover image)
IMAGE_PATH = "percorso/alla/tua/immagine.png"

# Messaggio segreto da nascondere (solo caratteri ASCII)
MESSAGE = "Messaggio segreto"

# Directory di output per i risultati
OUTPUT_DIR = "output"
```

### Parametri YOLO

```python
YOLO_MODEL = "yolov8n.pt"       # Modello YOLOv8
YOLO_CONFIDENCE = 0.25          # Soglia di confidenza
STRATEGY = "C"                  # Strategia ROI: 'A', 'B', 'C'
BOX_INDEX = None                # Indice bounding box (solo strategia A)
```

### Parametri Steganografia

```python
BLOCK_SIZE = 8                  # Dimensione blocchi SVD (4, 8, 16)
SV_RANGE = "mid"                # Quali SV modificare: 'first', 'mid', 'last'
DELTA = 15.0                    # Ampiezza spostamento additivo (5-30)
```

### Fasi da Eseguire

Abilita/disabilita le fasi impostando `True` o `False`:

```python
RUN_VALIDATION = True           # Fase 1: Validazione SVD
RUN_MEAN_CENTERING_TEST = False # Test mean centering
RUN_SVD_DEMO = True             # Fase 1: Demo SVD su immagine
RUN_YOLO = True                 # Fase 2: YOLO + ROI
RUN_EMBED = True                # Fase 3: Embedding messaggio
RUN_EXTRACT = True              # Fase 4: Estrazione (informed)
```

## 4. Esecuzione

Una volta configurato, esegui semplicemente:

```bash
python main.py
```

Le fasi abilitate verranno eseguite in sequenza.

## 5. Fase 1 — SVD from Scratch

### Solo Validazione

Per eseguire solo la validazione SVD custom vs numpy:

```python
RUN_VALIDATION = True
RUN_SVD_DEMO = False
RUN_YOLO = False
RUN_EMBED = False
RUN_EXTRACT = False
```

### Demo su Immagine

Per testare la ricostruzione di un'immagine:

```python
IMAGE_PATH = "mia_immagine.jpg"
RUN_VALIDATION = True
RUN_SVD_DEMO = True
```

I risultati verranno salvati nella cartella `output/`.

## 6. Fase 2 — Rilevamento YOLO e ROI

```python
IMAGE_PATH = "foto.png"
RUN_YOLO = True
```

### Strategie di Embedding

| Strategia          | Valore           | Descrizione                       |
| ------------------ | ---------------- | --------------------------------- |
| **A — Soggetti**   | `STRATEGY = "A"` | Nasconde dentro un bounding box   |
| **B — Sfondo**     | `STRATEGY = "B"` | Nasconde nello sfondo             |
| **C — Automatico** | `STRATEGY = "C"` | Bounding box più grande (default) |

### Output Fase 2

| File                       | Descrizione                           |
| -------------------------- | ------------------------------------- |
| `yolo_detections.png`      | Immagine con i bounding box disegnati |
| `roi_extracted.png`        | La ROI estratta                       |
| `roi_mask.png`             | Maschera binaria della ROI            |
| `roi_reconstructed_k*.png` | Ricostruzioni SVD della ROI           |

## 7. Fase 3 — Embedding del Messaggio

```python
IMAGE_PATH = "foto.png"
MESSAGE = "Messaggio segreto"       # Solo caratteri ASCII!
RUN_EMBED = True
```

Il messaggio viene codificato in **ASCII a 7 bit** (128 caratteri). Non sono supportati caratteri con accenti o simboli speciali fuori dal range ASCII (0-127).

### Parametri Avanzati di Embedding

| Parametro    | Valori                 | Effetto                              |
| ------------ | ---------------------- | ------------------------------------ |
| `BLOCK_SIZE` | `4`, `8`, `16`         | Blocchi più piccoli = più capacità   |
| `SV_RANGE`   | `first`, `mid`, `last` | Compromesso robustezza/invisibilità  |
| `DELTA`      | `5.0` - `30.0`         | Più alto = più robusto, più visibile |

### Trade-off SV Range

| Range   | Robustezza | Invisibilità | Consigliato per        |
| ------- | ---------- | ------------ | ---------------------- |
| `first` | ★★★        | ★            | Test di robustezza     |
| `mid`   | ★★         | ★★           | Uso generale (default) |
| `last`  | ★          | ★★★          | Massima invisibilità   |

### Output Fase 3

| File                  | Descrizione                          |
| --------------------- | ------------------------------------ |
| `stego_image.png`     | L'immagine con il messaggio nascosto |
| `yolo_detections.png` | Immagine con i bounding box YOLO     |

## 8. Fase 4 — Estrazione del Messaggio (Informed)

L'estrazione richiede **sia la stego-image che l'immagine originale**. Non è più un'estrazione blind.

```python
IMAGE_PATH = "foto.png"             # L'immagine ORIGINALE
RUN_EMBED = True                    # Crea la stego-image
RUN_EXTRACT = True                  # Estrai il messaggio
```

Quando `RUN_EXTRACT = True`, lo script:

1. Carica la stego-image da `output/stego_image.png`
2. Carica l'immagine originale da `IMAGE_PATH`
3. Rileva la ROI con YOLO
4. Confronta i valori singolari per estrarre i bit
5. Decodifica la sequenza ASCII 7-bit e stampa il messaggio

> ⚠️ **Importante**: i parametri `BLOCK_SIZE`, `SV_RANGE`, `DELTA` e `STRATEGY` devono corrispondere **esattamente** a quelli usati in fase di embedding.

### Troubleshooting

| Problema                          | Possibile causa                  | Soluzione                                          |
| --------------------------------- | -------------------------------- | -------------------------------------------------- |
| Messaggio vuoto o illeggibile     | Parametri non corrispondenti     | Usare gli stessi `SV_RANGE`, `BLOCK_SIZE`, `DELTA` |
| ROI diversa                       | Strategia diversa                | Usare la stessa `STRATEGY` e `YOLO_CONFIDENCE`     |
| Caratteri non-ASCII nel messaggio | Caratteri con accenti o speciali | Usare solo caratteri ASCII (0-127)                 |
| Stego-image non trovata           | Embedding non eseguito           | Impostare `RUN_EMBED = True`                       |
| Immagine originale non corretta   | PATH errato                      | Verificare `IMAGE_PATH`                            |

## 9. Struttura del Progetto

- `src/svd.py`: Implementazione core dell'algoritmo SVD tramite metodo delle potenze e deflazione.
- `src/image_utils.py`: Funzioni per conversione immagini, padding, blocchi e mean centering.
- `src/validation.py`: Logica di confronto tra l'implementazione custom e quella ufficiale di NumPy.
- `src/yolo_roi.py`: **[Fase 2]** Integrazione YOLOv8 per object detection e selezione della ROI con strategie A/B/C.
- `src/steganography.py`: **[Fase 3/4]** Embedding additivo e estrazione informed del payload, codifica ASCII 7-bit.
- `main.py`: Entry point del programma con configurazione diretta (Fasi 1–4).
