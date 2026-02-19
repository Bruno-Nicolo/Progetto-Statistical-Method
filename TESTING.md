# 🧪 Suite di Test — Valutazione Performance Steganografia SVD

Questo documento descrive la suite di test implementata in `tests/test_performance.py` per valutare le performance dell'algoritmo di steganografia al variare di tutti i parametri.

---

## Esecuzione

```bash
# Tutti i test con immagine sintetica (256×256)
python tests/test_performance.py

# Tutti i test con un'immagine reale
python tests/test_performance.py --image percorso/immagine.png

# Singolo test (es. solo il test 2)
python tests/test_performance.py --test 2

# Output in una directory specifica
python tests/test_performance.py --output risultati/
```

I risultati vengono stampati in console come tabelle formattate e salvati in file CSV nella directory di output (`test_output/` di default).

---

## Test Implementati

### Test 1 — Sweep Completo dei Parametri

**Obiettivo**: Valutare l'algoritmo su **tutte le combinazioni** di parametri.

| Parametro    | Valori testati                                    |
| ------------ | ------------------------------------------------- |
| `block_size` | 4, 8, 16                                          |
| `sv_range`   | first, mid, last                                  |
| `delta`      | 5, 10, 15, 20, 25, 30                             |
| Messaggio    | corto (6 char), medio (54 char), lungo (120 char) |

**Metriche misurate**: PSNR, SSIM, BER, correttezza, tempo di esecuzione.  
**Combinazioni totali**: 3 × 3 × 6 × 3 = **162 test**.  
**Output CSV**: `results_sweep.csv`

---

### Test 2 — Impatto del Delta sulla Qualità Visiva

**Obiettivo**: Trovare il **trade-off ottimale** tra robustezza e invisibilità variando delta con granularità fine.

| Parametro  | Valori fissi                              |
| ---------- | ----------------------------------------- |
| Block size | 8                                         |
| SV range   | mid                                       |
| Delta      | 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50 |

**Cosa si aspetta**: Al crescere di delta, il PSNR decresce (alterazione più visibile) ma la robustezza aumenta.  
**Output CSV**: `results_delta.csv`

---

### Test 3 — Analisi della Capacità

**Obiettivo**: Calcolare la **capacità massima** (in bit e caratteri) per ogni combinazione di `block_size` e `sv_range`, e verificare che il messaggio venga estratto correttamente al limite della capacità.

**Cosa si aspetta**:

- `block_size=4` offre la massima capacità (più blocchi per unità di area)
- `sv_range='first'` o `'last'` offrono circa ⅓ dei SV per blocco
- `sv_range='mid'` offre circa ⅓ dei SV per blocco

**Output CSV**: `results_capacity.csv`

---

### Test 4 — Robustezza al Rumore Gaussiano

**Obiettivo**: Misurare la **resistenza del messaggio** quando si aggiunge rumore all'immagine stego.

| Parametro | Valori testati                   |
| --------- | -------------------------------- |
| Noise σ   | 0, 1, 2, 3, 5, 8, 10, 15, 20, 30 |
| Delta     | 10, 15, 25 (per confronto)       |

**Procedura**:

1. Embedding con parametri fissi (block_size=8, sv_range=mid)
2. Aggiunta di rumore gaussiano alla stego-image
3. Tentativo di estrazione dalla stego-image rumorosa
4. Misurazione BER e correttezza

**Cosa si aspetta**: Delta più grandi resistono meglio al rumore.  
**Output CSV**: `results_noise.csv`

---

### Test 5 — Robustezza al Blur Gaussiano

**Obiettivo**: Misurare la resistenza del messaggio al **blur** (sfocatura).

| Parametro   | Valori testati |
| ----------- | -------------- |
| Blur radius | 0, 1, 2, 3, 5  |
| Delta       | 10, 15, 25     |

**Procedura**: Simile al Test 4, ma con blur invece di rumore.  
**Cosa si aspetta**: Il blur è particolarmente distruttivo per i dettagli ad alta frequenza (ultimi SV).  
**Output CSV**: `results_blur.csv`

---

### Test 6 — Robustezza alla Compressione JPEG

**Obiettivo**: Verificare se il messaggio sopravvive alla **compressione lossy JPEG**.

| Parametro    | Valori testati              |
| ------------ | --------------------------- |
| JPEG quality | 100, 95, 90, 80, 70, 50, 30 |
| Delta        | 10, 15, 25                  |

**Procedura**:

1. Embedding nella stego-image (PNG, lossless)
2. Compressione in JPEG e ricaricamento
3. Tentativo di estrazione dalla versione JPEG
4. Misurazione BER e correttezza

**Cosa si aspetta**: La compressione JPEG è il test più aggressivo. Solo delta molto alti potrebbero resistere a qualità basse.  
**Output CSV**: `results_jpeg.csv`

---

### Test 7 — Impatto Visivo dei SV Range

**Obiettivo**: **Confronto visivo** tra le tre strategie di selezione dei valori singolari.

Per ogni `sv_range` (first, mid, last), con block_size=8 e delta=15:

- Salva la stego-image
- Genera una **mappa delle differenze** amplificata (10×) per evidenziare le alterazioni

**Output**:
| File | Descrizione |
|-------------------------|---------------------------------------|
| `stego_first.png` | Stego-image con SV first |
| `stego_mid.png` | Stego-image con SV mid |
| `stego_last.png` | Stego-image con SV last |
| `diff_first.png` | Differenza amplificata 10× (first) |
| `diff_mid.png` | Differenza amplificata 10× (mid) |
| `diff_last.png` | Differenza amplificata 10× (last) |

**Metriche**: PSNR, SSIM, differenza massima e media.  
**Output CSV**: `results_sv_visual.csv`

---

### Test 8 — Scalabilità con Lunghezza del Messaggio

**Obiettivo**: Misurare come le performance **scalano** al crescere della lunghezza del messaggio.

| Parametro     | Valori testati                       |
| ------------- | ------------------------------------ |
| Lunghezza msg | 1, 5, 10, 20, 50, 100, max chars     |
| Parametri     | block_size=8, sv_range=mid, delta=15 |

**Cosa si aspetta**: Al crescere del messaggio, il PSNR decresce gradualmente e il tempo di esecuzione resta quasi costante (l'SVD si calcola comunque su tutti i blocchi).  
**Output CSV**: `results_scaling.csv`

---

## Metriche

| Metrica         | Descrizione                 | Range     | Ideale                   |
| --------------- | --------------------------- | --------- | ------------------------ |
| **PSNR**        | Peak Signal-to-Noise Ratio  | 0 – ∞ dB  | > 40 dB (impercettibile) |
| **SSIM**        | Structural Similarity Index | 0.0 – 1.0 | > 0.99                   |
| **BER**         | Bit Error Rate              | 0.0 – 1.0 | 0.0 (nessun errore)      |
| **Correttezza** | Match esatto del messaggio  | ✅/❌     | ✅                       |

### Interpretazione PSNR

| PSNR (dB) | Qualità                                  |
| --------- | ---------------------------------------- |
| > 50      | Alterazioni invisibili, qualità perfetta |
| 40 – 50   | Ottimo, impercettibile all'occhio        |
| 30 – 40   | Buono, lievi artefatti possibili         |
| 20 – 30   | Mediocre, alterazioni visibili           |
| < 20      | Scarso, alterazioni evidenti             |

---

## Output

Tutti i risultati vengono salvati nella directory `test_output/` (configurabile con `--output`):

| File                       | Contenuto                                           |
| -------------------------- | --------------------------------------------------- |
| `results_sweep.csv`        | Risultati di tutte le 162 combinazioni di parametri |
| `results_delta.csv`        | Impatto del delta (12 valori)                       |
| `results_capacity.csv`     | Capacità per ogni configurazione (9 combinazioni)   |
| `results_noise.csv`        | Robustezza al rumore (30 test)                      |
| `results_blur.csv`         | Robustezza al blur (15 test)                        |
| `results_jpeg.csv`         | Robustezza JPEG (21 test)                           |
| `results_sv_visual.csv`    | Impatto visivo dei 3 SV range                       |
| `results_scaling.csv`      | Scalabilità con lunghezza messaggio                 |
| `stego_*.png`              | Stego-image per confronto visivo                    |
| `diff_*.png`               | Mappe di differenza amplificate                     |
| `test_image_synthetic.png` | Immagine sintetica usata (se non fornita)           |
