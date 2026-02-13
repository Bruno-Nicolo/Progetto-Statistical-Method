# Guida all'Esecuzione

Questo documento fornisce le istruzioni per configurare l'ambiente ed eseguire il modulo di **SVD from scratch** e la relativa suite di validazione.

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

## 3. Esecuzione del Codice

Il file principale è `main.py`, che funge da interfaccia CLI per tutte le funzionalità implementate finora.

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

### Opzioni Avanzate

Visualizza tutti i parametri disponibili:

```bash
python main.py --help
```

- `--output`, `-o`: Specifica la directory per le immagini ricostruite (default: `output`).
- `--no-validate`: Salta i test matematici iniziali e passa direttamente alla demo immagine.

## 4. Struttura del Progetto

- `src/svd.py`: Implementazione core dell'algoritmo SVD tramite metodo delle potenze e deflazione.
- `src/image_utils.py`: Funzioni per conversione immagini, padding, blocchi e mean centering.
- `src/validation.py`: Logica di confronto tra l'implementazione custom e quella ufficiale di NumPy.
- `main.py`: Entry point del programma.
