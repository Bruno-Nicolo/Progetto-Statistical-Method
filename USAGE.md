# Guida all'Esecuzione

Questo documento fornisce le istruzioni per configurare l'ambiente ed eseguire il modulo di **SVD from scratch** (Fase 1), il **rilevamento YOLO con selezione della ROI** (Fase 2), l'**embedding del messaggio** (Fase 3) e l'**estrazione del messaggio** dalla stego-image (Fase 4).

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

## 5. Fase 3 — Embedding del Messaggio

La Fase 3 combina YOLO (ROI) + SVD custom per nascondere un messaggio di testo all'interno dell'immagine.

### Embedding Base

```bash
python main.py --image foto.png --embed --message "Messaggio segreto"
```

Questo comando:
1. Rileva oggetti con YOLO e seleziona la ROI (strategia C di default)
2. Converte il messaggio in sequenza binaria
3. Suddivide la ROI in blocchi 8×8 e applica la SVD custom
4. Modifica i valori singolari intermedi con QIM (Quantization Index Modulation)
5. Ricostruisce la stego-image e la salva
6. Verifica l'embedding estraendo immediatamente il messaggio

### Scelta dei Valori Singolari (flag `--sv-range`)

Il flag `--sv-range` permette di testare quale range di valori singolari alterare:

| Range | Flag | Comportamento | Trade-off |
|-------|------|---------------|-----------|
| **Primi** | `--sv-range first` | Modifica i primi ~1/3 dei SV | Massima robustezza, ma artefatti visibili |
| **Intermedi** | `--sv-range mid` | Modifica i SV centrali (default) | Miglior compromesso tra invisibilità e robustezza |
| **Ultimi** | `--sv-range last` | Modifica gli ultimi ~1/3 dei SV | Invisibile all'occhio, ma vulnerabile a JPEG |

**Esempi per il test comparativo:**

```bash
# Test con i primi valori singolari (più robusto)
python main.py --image foto.png --embed --message "Test" --sv-range first

# Test con i valori intermedi (compromesso, default)
python main.py --image foto.png --embed --message "Test" --sv-range mid

# Test con gli ultimi valori singolari (più invisibile)
python main.py --image foto.png --embed --message "Test" --sv-range last
```

### Parametri di Embedding Avanzati

```bash
# Blocchi più grandi (16×16) per immagini di alta risoluzione
python main.py --image foto.png --embed --message "Test" --block-size 16

# Delta più alto = più robusto ma alterazioni più evidenti
python main.py --image foto.png --embed --message "Test" --delta 25

# Delta più basso = meno visibile ma più fragile
python main.py --image foto.png --embed --message "Test" --delta 5

# Combinazione completa con strategia A e parametri custom
python main.py --image foto.png --embed --message "Top Secret" \
    --strategy A --sv-range mid --block-size 8 --delta 15
```

### Output Fase 3

La Fase 3 salva nella cartella `output/`:

| File | Descrizione |
|------|-------------|
| `stego_image.png` | L'immagine con il messaggio nascosto |
| `yolo_detections.png` | Immagine con i bounding box YOLO evidenziati |

Il report in console mostra anche:
- **Capacità** della ROI (quanti bit/caratteri si possono nascondere)
- **PSNR** (Peak Signal-to-Noise Ratio) tra cover e stego-image
- **Verifica** immediata dell'estrazione del messaggio

## 6. Fase 4 — Estrazione del Messaggio

La Fase 4 esegue il processo inverso all'embedding: recupera il messaggio nascosto dalla stego-image **senza bisogno dell'immagine originale** (estrazione blind).

### Estrazione Base

```bash
python main.py --image output/stego_image.png --extract
```

Questo comando:
1. Ripassa la stego-image sotto YOLO per ritrovare le coordinate della ROI
2. Suddivide la ROI in blocchi e applica la SVD custom ad ogni blocco
3. Estrae i bit dal quoziente di quantizzazione dei valori singolari (QIM blind)
4. Decodifica la sequenza binaria in testo e stampa il messaggio recuperato

### Parametri di Estrazione

> ⚠️ **Importante**: i parametri `--block-size`, `--sv-range`, `--delta` e `--strategy` devono corrispondere **esattamente** a quelli usati in fase di embedding. Se anche uno solo è diverso, il messaggio non sarà recuperabile.

```bash
# Estrazione con parametri di default (block-size=8, sv-range=mid, delta=15)
python main.py --image output/stego_image.png --extract

# Se l'embedding è stato fatto con sv-range first e delta 20
python main.py --image stego.png --extract --sv-range first --delta 20

# Se l'embedding è stato fatto con strategia A e blocchi 16×16
python main.py --image stego.png --extract --strategy A --block-size 16

# Combinazione completa dei parametri
python main.py --image stego.png --extract \
    --strategy C --sv-range mid --block-size 8 --delta 15
```

### Output Fase 4

Il messaggio estratto viene stampato direttamente nel terminale. In aggiunta:

| File | Descrizione |
|------|-------------|
| `extract_yolo_detections.png` | Stego-image con i bounding box YOLO evidenziati (verifica che la ROI sia la stessa) |

### Troubleshooting

Se l'estrazione non recupera il messaggio corretto, verificare:

| Problema | Possibile causa | Soluzione |
|----------|----------------|-----------|
| Messaggio vuoto o illeggibile | Parametri non corrispondenti | Usare gli stessi `--sv-range`, `--block-size`, `--delta` dell'embedding |
| ROI diversa | Strategia diversa o YOLO rileva box differenti | Usare la stessa `--strategy` e `--yolo-confidence` |
| Caratteri corrotti | Immagine compressa dopo l'embedding | Assicurarsi che la stego-image sia in formato PNG (lossless) |

## 7. Opzioni Avanzate

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
| `--strategy`, `-s` | Strategia di embedding/estrazione: A, B, C (default: C) |
| `--box-index` | Indice del bounding box (solo strategia A) |
| `--yolo-model` | Modello YOLOv8 (default: `yolov8n.pt`) |
| `--yolo-confidence` | Soglia di confidenza YOLO (default: 0.25) |
| `--embed` | Attiva l'embedding del messaggio (Fase 3) |
| `--message`, `-m` | Il messaggio segreto da nascondere |
| `--extract` | Attiva l'estrazione del messaggio (Fase 4) |
| `--sv-range` | Quali SV modificare/leggere: `first`, `mid`, `last` (default: `mid`) |
| `--block-size` | Dimensione blocchi SVD: 4, 8, 16 (default: 8) |
| `--delta` | Passo di quantizzazione QIM (default: 15.0) |

## 8. Struttura del Progetto

- `src/svd.py`: Implementazione core dell'algoritmo SVD tramite metodo delle potenze e deflazione.
- `src/image_utils.py`: Funzioni per conversione immagini, padding, blocchi e mean centering.
- `src/validation.py`: Logica di confronto tra l'implementazione custom e quella ufficiale di NumPy.
- `src/yolo_roi.py`: **[Fase 2]** Integrazione YOLOv8 per object detection e selezione della ROI con strategie A/B/C.
- `src/steganography.py`: **[Fase 3/4]** Embedding e estrazione blind del payload tramite SVD + QIM (Quantization Index Modulation).
- `main.py`: Entry point del programma con CLI completa (Fasi 1–4).
