# 🧪 Suite di Test — Steganografia SVD

Questo progetto include due suite di test complementari: una per la verifica rapida della correttezza funzionale (`tests/quick_test.py`) e una per l'analisi approfondita delle performance e della robustezza (`tests/test_performance.py`).

---

## 1. Quick Test (`tests/quick_test.py`)

**Scopo**: Verificare in pochi secondi che tutti i componenti fondamentali del sistema funzionino correttamente dal punto di vista matematico e logico. È il test da lanciare dopo ogni modifica al codice (Regression Testing).

### Esecuzione

```bash
python tests/quick_test.py
```

### Funzionamento Dettagliato

Lo script esegue 7 batterie di test su dati sintetici generati casualmente (piccole matrici) per garantire la massima velocità:

1.  **TEST 1 — Correttezza SVD**: Confronta l'implementazione custom di `svd_compact` con `np.linalg.svd`. Verifica l'errore sui valori singolari, l'errore di ricostruzione della matrice originale e l'ortogonalità delle matrici U e V.
2.  **TEST 2 — Conversione Testo ↔ Binario**: Verifica che la trasformazione delle stringhe in bit e viceversa sia reversibile senza perdite (bit-perfect).
3.  **TEST 3 — Roundtrip Embedding/Extraction**: Esegue cicli completi di inserimento ed estrazione su matrici 32×32 con diverse combinazioni di parametri (block size, range, delta), verificando che il messaggio estratto coincida con l'originale.
4.  **TEST 4 — Qualità Visiva (Coarse)**: Controlla che il PSNR non scenda sotto soglie critiche per diversi valori di delta.
5.  **TEST 5 — Edge Cases**: Verifica il comportamento con immagini uniformi (tutte grigie), valori estremi (0 o 255) e matrici con dimensioni non multiple del block size.
6.  **TEST 6 — Consistenza SV Range**: Assicura che le tre strategie di selezione (`first`, `mid`, `last`) siano tutte funzionanti e coerenti.
7.  **TEST 7 — Selezione ROI**: Testa la logica di YOLO/ROI (class `BoundingBox` e strategie A, B, C) e verifica che i pixel esterni alla ROI non vengano minimamente modificati durante l'embedding.

---

## 2. Performance Test (`tests/test_performance.py`)

**Scopo**: Valutare le performance dell'algoritmo al variare di tutti i parametri critici e misurarne la robustezza contro attacchi comuni o degradazioni dell'immagine.

### Esecuzione

```bash
# Test standard con immagine sintetica (64×64 o 256×256)
python tests/test_performance.py

# Test con un'immagine reale specifica
python tests/test_performance.py --image percorso/immagine.png

# Esegui solo un test specifico (es. Test 4 sul rumore)
python tests/test_performance.py --test 4 --image percorso/immagine.png

# Specifica la directory per i file CSV e le immagini di output
python tests/test_performance.py --output risultati_analisi/
```

### Test Implementati in Dettaglio

#### Test 1 — Sweep Completo dei Parametri

Valuta l'algoritmo su **162 combinazioni** (3 block_sizes × 3 sv_ranges × 6 deltas × 3 lunghezze messaggio). Identifica quali configurazioni falliscono per capacità insufficiente e calcola le medie globali di PSNR e SSIM.

#### Test 2 — Impatto del Delta

Analizza con granularità fine (12 valori da 1.0 a 50.0) come varia la qualità visiva. È fondamentale per tracciare la curva di trade-off tra invisibilità e robustezza.

#### Test 3 — Analisi della Capacità

Calcola il numero massimo di bit e caratteri inseribili per ogni configurazione di blocco. Verifica inoltre che l'estrazione sia corretta quando si riempie la ROI al 100% della sua capacità.

#### Test 4, 5, 6 — Robustezza (Rumore, Blur, JPEG)

Sono i "stress test" del sistema. Applicano distorsioni all'immagine stego prima dell'estrazione:

- **Rumore Gaussiano**: Varie intensità di deviazione standard ($\sigma$).
- **Blur Gaussiano**: Diversi raggi di sfocatura.
- **Compressione JPEG**: Diversi fattori di qualità (da 100 a 30).
  Misura il **BER (Bit Error Rate)** per quantificare quanti bit vengono persi a causa del degrado.

#### Test 7 — Confronto Visivo SV Range

Genera immagini di differenza amplificata (10×) per mostrare graficamente _dove_ l'algoritmo va a modificare i pixel a seconda che si scelgano i valori singolari iniziali, centrali o finali.

#### Test 8 — Scalabilità Messaggio

Verifica come variano le metriche all'aumentare del carico (da 1 carattere fino al limite massimo della ROI).

#### Test 9 — ROI Multi-dimensione

Valuta l'inserimento dello stesso messaggio in ROI di diverse dimensioni (100%, 75%, 50%, 25% dell'immagine) e verifica il corretto utilizzo della Strategia B (sfondo).

#### Test 10 — Metriche di Approssimazione SVD

Analizza l'impatto dello steganogramma sull'algebra lineare sottostante, computando per ogni configurazione `(block_size, sv_range, delta)`:

1.  **Errore di Approssimazione**:
    -   **Norma di Frobenius** $\|A - A_{stego}\|_F$: misura l'errore quadratico totale della matrice differenza. Equivale a $\sqrt{\sum_{i,j}(a_{ij} - a'_{ij})^2}$.
    -   **Norma Spettrale** (2-norma) $\|A - A_{stego}\|_2 = \sigma_{\max}(A - A_{stego})$: quantifica il massimo "allungamento" causato dalla modifica steganografica. È il primo valore singolare della matrice differenza.
2.  **Fattore di Compressione e Occupazione di Memoria**:
    -   Fattore di compressione: $2k\left(\frac{1}{m} + \frac{1}{n}\right)$, dove $k$ è il rango del blocco e $m \times n$ è la sua dimensione.
    -   Confronto scalari: matrice piena ($m \times n$) vs. rappresentazione SVD troncata ($k(m + n + 1)$).
3.  **Numero di Condizionamento**:
    -   $\kappa(A) = \frac{\sigma_1}{\sigma_n}$ calcolato per-blocco e mediato, sia sulla ROI originale ($\kappa_{orig}$) che sulla ROI stego ($\kappa_{stego}$). Un aumento significativo di $\kappa$ segnala che l'embedding ha reso la matrice più mal condizionata.

### Output dei Test

Tutti i risultati vengono salvati nella directory `test_output/` (configurabile con `--output`):

| File                       | Contenuto                                           |
| :------------------------- | :-------------------------------------------------- |
| `results_sweep.csv`        | Risultati di tutte le 162 combinazioni di parametri |
| `results_delta.csv`        | Impatto del delta sulla qualità (Test 2)            |
| `results_capacity.csv`     | Analisi capacità massima (Test 3)                   |
| `results_noise.csv`        | Robustezza al rumore (Test 4)                       |
| `results_blur.csv`         | Robustezza al blur (Test 5)                         |
| `results_jpeg.csv`         | Robustezza alla compressione JPEG (Test 6)          |
| `results_sv_visual.csv`    | Metriche del confronto visivo (Test 7)              |
| `results_scaling.csv`      | Scalabilità con lunghezza messaggio (Test 8)        |
| `results_approx_metrics.csv` | Norme, compressione e condizionamento (Test 10)   |
| `stego_*.png`              | Immagini stego per confronto visivo                 |
| `diff_*.png`               | Mappe di differenza amplificate 10×                 |
| `test_image_synthetic.png` | Immagine sintetica usata (se non fornita)           |

### Metriche di Valutazione

| Metrica         | Descrizione                                       | Range     | Target  |
| :-------------- | :------------------------------------------------ | :-------- | :------ |
| **PSNR**        | Peak Signal-to-Noise Ratio (fedeltà cromatica)    | 0 – ∞ dB  | > 40 dB |
| **SSIM**        | Structural Similarity Index (fedeltà strutturale) | 0.0 – 1.0 | > 0.99  |
| **BER**         | Bit Error Rate (percentuale bit errati)           | 0.0 – 1.0 | 0.0     |
| **Correttezza** | Match esatto del messaggio stringa                | ✅/❌     | ✅      |
| **$\|\cdot\|_F$** | Norma di Frobenius di $A - A_{stego}$           | 0 – ∞     | → 0     |
| **$\|\cdot\|_2$** | Norma Spettrale (2-norma) di $A - A_{stego}$    | 0 – ∞     | → 0     |
| **CF**          | Fattore di compressione $2k(1/m+1/n)$             | 0 – ∞     | < 1     |
| **$\kappa$**    | Numero di condizionamento $\sigma_1/\sigma_n$     | 1 – ∞     | → 1     |

---

## 3. Confronto tra Quick Test e Performance Test

| Caratteristica           | `quick_test.py`                 | `test_performance.py`                     |
| :----------------------- | :------------------------------ | :---------------------------------------- |
| **Obiettivo Principale** | Correttezza del codice          | Analisi scientifica e robustezza          |
| **Dati in Input**        | Matrici random molto piccole    | Immagini reali o sintetiche strutturate   |
| **Velocità**             | Ultra-rapido (< 2 secondi)      | Lento (dipende dal numero di test)        |
| **Output**               | Log test Passati/Falliti        | Tabelle, file CSV, immagini di differenza |
| **Uso in Sviluppo**      | Da usare durante il coding (CI) | Da usare per calibrare i parametri        |
| **Metriche**             | Solo base (Match, Errore SVD)   | Avanzate (SSIM, BER, norme, $\kappa$, curve di robustezza) |
| **Ambito**               | Unit Testing                    | Benchmarking / Stress Testing             |

In sintesi: usa `quick_test.py` per essere sicuro di non aver rotto nulla; usa `test_performance.py` per capire quanto è "buona" una specifica configurazione di parametri.
