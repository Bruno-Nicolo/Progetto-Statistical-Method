# Documentazione Grafici — Steganografia SVD

Questo documento descrive i grafici generati dallo script `tests/plot_results.py` a partire dai CSV aggregati prodotti da `tests/test_dataset.py`.

I grafici vengono salvati nella cartella `test_output/plots/`.

---

## Indice

1. [Test 1 — Sweep Parametri](#test-1--sweep-parametri)
2. [Test 2 — Impatto del Delta](#test-2--impatto-del-delta)
3. [Test 3 — Robustezza al Rumore](#test-3--robustezza-al-rumore)
4. [Test 4 — Robustezza al Blur](#test-4--robustezza-al-blur)
5. [Test 5 — Robustezza alla Compressione JPEG](#test-5--robustezza-alla-compressione-jpeg)
6. [Test 6 — Confronto Strategie ROI](#test-6--confronto-strategie-roi)
7. [Grafici Riassuntivi Cross-Test](#grafici-riassuntivi-cross-test)

---

## Metriche Utilizzate

| Metrica | Descrizione | Valore ideale |
|---------|-------------|---------------|
| **PSNR** (Peak Signal-to-Noise Ratio) | Misura la qualità dell'immagine stego rispetto all'originale, in dB. Valori più alti indicano minore distorsione. | > 40 dB (impercettibile) |
| **SSIM** (Structural Similarity Index) | Misura la similarità strutturale percettiva tra immagine originale e stego. Varia tra 0 e 1. | → 1.0 |
| **BER** (Bit Error Rate) | Percentuale di bit estratti erroneamente rispetto ai bit originali del messaggio. | → 0.0 |
| **Correct%** | Percentuale di immagini in cui il messaggio è stato estratto correttamente al 100%. | → 100% |

---

## Test 1 — Sweep Parametri

**Sorgente CSV:** `agg_sweep.csv`

Questo test varia sistematicamente tutti i parametri di embedding: `block_size` (4, 8, 16), `sv_range` (first, mid, last), `delta` (5–30) e lunghezza del messaggio (corto, medio, lungo).

### `sweep_heatmap_delta<N>.png`

**Tipo:** Heatmap

**Assi:** asse X = `sv_range` (first, mid, last), asse Y = `block_size` (4, 8, 16)

**Colore:** PSNR medio (dB)

**Cosa mostra:** Per un dato valore di delta, qual è la combinazione di `block_size` e `sv_range` che produce la migliore qualità visiva (PSNR più alto). Viene generata una heatmap per ciascun valore di delta testato.

**Come interpretarlo:**
- Celle più scure (valori alti) → minor distorsione → configurazione preferibile per la qualità visiva.
- Generalmente, block size più grandi e `sv_range = mid` tendono a dare PSNR migliori perché distribuiscono il messaggio in modo più uniforme.

---

### `sweep_psnr_vs_delta_bar.png`

**Tipo:** Bar chart raggruppato

**Asse X:** Valori di delta (δ)

**Asse Y:** PSNR medio (dB)

**Gruppi:** Una barra per ciascun `sv_range` (first, mid, last)

**Cosa mostra:** Come il PSNR degrada all'aumentare del delta, e quale `sv_range` mantiene la qualità migliore. Usa il messaggio "corto" e media su tutti i block size.

**Come interpretarlo:**
- All'aumentare del delta, il PSNR diminuisce (più distorsione).
- Se un `sv_range` ha barre costantemente più alte, è il più adatto a preservare la qualità dell'immagine.

---

### `sweep_psnr_vs_ber_scatter.png`

**Tipo:** Scatter plot

**Asse X:** PSNR medio (dB)

**Asse Y:** BER medio

**Colore:** `sv_range`

**Cosa mostra:** Il **trade-off** fondamentale tra qualità visiva (PSNR) e accuratezza dell'estrazione (BER). Ogni punto rappresenta una configurazione (block_size, sv_range, delta, messaggio).

**Come interpretarlo:**
- L'area ideale è **in alto a sinistra** (PSNR alto, BER basso).
- Punti con PSNR alto ma BER alto indicano un delta troppo basso: l'immagine è bella ma il messaggio non sopravvive.
- Punti con BER = 0 ma PSNR basso indicano un delta troppo alto: il messaggio è perfetto ma l'immagine è visibilmente alterata.
- La "frontiera di Pareto" del trade-off aiuta a scegliere il delta ottimale.

---

## Test 2 — Impatto del Delta

**Sorgente CSV:** `agg_delta.csv`

Questo test isola l'effetto del parametro delta (da 1.0 a 50.0) sulla qualità e sull'accuratezza, mantenendo fissi `block_size=8` e `sv_range=mid`.

### `delta_psnr.png`

**Tipo:** Line plot con banda d'errore (±1 deviazione standard)

**Asse X:** Delta (δ)

**Asse Y:** PSNR (dB)

**Cosa mostra:** La curva di degradazione del PSNR all'aumentare di delta. La banda colorata indica la variabilità tra le diverse immagini del dataset.

**Come interpretarlo:**
- La curva è tipicamente decrescente: delta più alti → più distorsione → PSNR più basso.
- La banda stretta indica un comportamento coerente tra immagini diverse; una banda larga indica che alcune immagini sono più sensibili al delta.

---

### `delta_ssim.png`

**Tipo:** Line plot con banda d'errore (±1σ)

**Asse X:** Delta (δ)

**Asse Y:** SSIM

**Cosa mostra:** L'evoluzione della similarità strutturale (SSIM) al variare del delta. L'SSIM cattura aspetti percettivi non colti dal PSNR.

**Come interpretarlo:**
- Un SSIM > 0.95 è generalmente considerato "impercettibile".
- L'SSIM tende a scendere più lentamente del PSNR per delta bassi, perché le alterazioni SVD iniziali non modificano fortemente la struttura.

---

### `delta_ber.png`

**Tipo:** Line plot

**Asse X:** Delta (δ)

**Asse Y:** BER

**Cosa mostra:** Come il BER cambia con delta, senza attacchi (condizioni ideali). Il BER dovrebbe essere 0 per tutti i delta se l'embedding/estrazione funziona correttamente.

**Come interpretarlo:**
- BER > 0 per delta bassi indica che il segnale è troppo debole per essere estratto in modo affidabile anche senza attacchi.
- La soglia di delta sotto la quale il BER inizia a crescere indica il **delta minimo pratico**.

---

### `delta_psnr_correct_dual.png`

**Tipo:** Line plot con doppio asse Y

**Asse X:** Delta (δ)

**Asse Y sinistro:** PSNR (dB)

**Asse Y destro:** Correct (%)

**Cosa mostra:** Il grafico sovrappone qualità visiva e accuratezza sullo stesso asse X, rendendo immediato il compromesso tra le due metriche.

**Come interpretarlo:**
- Il **delta ottimale** si trova dove la curva Correct% raggiunge il 100% e il PSNR è ancora accettabilmente alto.
- Se le due curve si incrociano, quel punto approssima il miglior compromesso.

---

## Test 3 — Robustezza al Rumore

**Sorgente CSV:** `agg_noise.csv`

Questo test aggiunge rumore gaussiano (sigma: 0–30) all'immagine stego prima dell'estrazione, per tre valori di delta (10, 15, 25).

### `noise_ber_vs_sigma.png`

**Tipo:** Line plot multiplo con bande d'errore (±1σ)

**Asse X:** Sigma del rumore gaussiano

**Asse Y:** BER medio

**Linee:** Una per ciascun delta testato

**Cosa mostra:** Come il BER cresce all'aumentare dell'intensità del rumore, e quanto un delta più alto protegge il messaggio.

**Come interpretarlo:**
- Delta alti (es. 25) producono curve che crescono più lentamente → il messaggio è più robusto.
- La soglia di sigma alla quale il BER supera 0.1 (10% di errori) indica il **limite di tolleranza al rumore** per quella configurazione.

---

### `noise_heatmap_ber.png`

**Tipo:** Heatmap

**Asse X:** Sigma del rumore

**Asse Y:** Delta (δ)

**Colore:** BER medio (scala rosso-verde: verde = basso, rosso = alto)

**Cosa mostra:** Una mappa 2D che sintetizza BER per ogni combinazione (delta, sigma). Permette di individuare rapidamente la "zona sicura" (verde).

**Come interpretarlo:**
- L'angolo in alto a sinistra (delta alto, sigma basso) è la zona più verde → condizioni ottimali.
- La transizione da verde a rosso segna il confine di robustezza del sistema.

---

### `noise_correct_pct.png`

**Tipo:** Bar chart raggruppato

**Asse X:** Sigma del rumore

**Asse Y:** Correct (%)

**Gruppi:** Un colore per ciascun delta

**Cosa mostra:** La percentuale di immagini in cui il messaggio viene estratto perfettamente, in funzione del rumore.

**Come interpretarlo:**
- Un calo brusco da 100% a 0% indica che il sistema non è graduale ma ha una **soglia critica** di rumore.
- Se la curva scende gradualmente, il sistema degrada in modo prevedibile.

---

## Test 4 — Robustezza al Blur

**Sorgente CSV:** `agg_blur.csv`

Questo test applica un blur gaussiano con raggi crescenti (0–5) all'immagine stego.

### `blur_ber_vs_radius.png`

**Tipo:** Line plot multiplo con bande d'errore (±1σ)

**Asse X:** Raggio del blur gaussiano

**Asse Y:** BER medio

**Linee:** Una per ciascun delta testato

**Cosa mostra:** Come il BER aumenta col raggio del blur. Il blur è un attacco particolarmente distruttivo per la steganografia SVD perché altera i valori singolari.

**Come interpretarlo:**
- Confrontare le curve: un delta più alto dovrebbe fornire maggiore resistenza.
- Se anche il delta più alto mostra un BER elevato con blur piccolo, il sistema è **vulnerabile al blur**.

---

### `blur_correct_pct.png`

**Tipo:** Bar chart raggruppato

**Asse X:** Raggio del blur

**Asse Y:** Correct (%)

**Gruppi:** Un colore per ciascun delta

**Cosa mostra:** La percentuale di estrazione corretta al variare del raggio di blur.

**Come interpretarlo:**
- Il blur tipicamente degrada le prestazioni molto più rapidamente rispetto al rumore gaussiano, poiché agisce direttamente sulle frequenze spaziali che l'SVD codifica.

---

## Test 5 — Robustezza alla Compressione JPEG

**Sorgente CSV:** `agg_jpeg.csv`

Questo test applica compressione JPEG con qualità variabile (100–30) all'immagine stego.

### `jpeg_ber_vs_quality.png`

**Tipo:** Line plot multiplo con bande d'errore (±1σ)

**Asse X:** Qualità JPEG (da 100 a 30, ordine decrescente → degradazione crescente)

**Asse Y:** BER medio

**Linee:** Una per ciascun delta testato

**Cosa mostra:** Come la compressione JPEG degrada il messaggio nascosto. L'asse X è invertito per mostrare la degradazione crescente da sinistra a destra.

**Come interpretarlo:**
- La compressione JPEG è l'attacco più comune nella distribuzione reale delle immagini.
- Un sistema robusto dovrebbe mantenere BER ≈ 0 per qualità ≥ 70.
- Delta alti migliorano la robustezza a scapito della qualità visiva.

---

### `jpeg_correct_pct.png`

**Tipo:** Bar chart raggruppato

**Asse X:** Qualità JPEG

**Asse Y:** Correct (%)

**Gruppi:** Un colore per ciascun delta

**Cosa mostra:** La percentuale di messaggi estratti correttamente per ogni livello di compressione JPEG.

**Come interpretarlo:**
- La soglia di qualità JPEG alla quale Correct% scende sotto il 100% è un indicatore pratico di quanto il sistema è utilizzabile in scenari reali (social media, web, ecc.).

---

## Test 6 — Confronto Strategie ROI

**Sorgente CSV:** `agg_roi.csv`

Questo test confronta 4 strategie di selezione della ROI (Region of Interest) usando YOLO:
- **A_soggetto:** embedding nel bounding box del soggetto rilevato
- **B_sfondo:** embedding nell'intera immagine (background-oriented)
- **C_automatico:** embedding nel box più grande rilevato
- **full_image:** embedding sull'intera immagine senza YOLO (baseline)

### `roi_comparison_bar.png`

**Tipo:** 4 bar chart affiancati (PSNR, SSIM, BER, Correct%)

**Asse X:** Strategia

**Asse Y:** Valore della metrica

**Cosa mostra:** Un confronto diretto tra le 4 strategie su tutte le metriche principali.

**Come interpretarlo:**
- **PSNR/SSIM:** Strategie che usano una ROI più piccola (es. A_soggetto) dovrebbero avere PSNR più alto sull'immagine complessiva, perché la distorsione è concentrata.
- **BER/Correct%:** La strategia migliore ha BER più basso e Correct% più alto.
- Se full_image ha prestazioni simili alle strategie ROI, l'uso di YOLO non aggiunge valore significativo.

---

### `roi_radar.png`

**Tipo:** Radar chart (Spider chart)

**Assi radiali:** PSNR, SSIM, Correct%, Outside OK% (normalizzati tra 0 e 1)

**Aree colorate:** Una per ciascuna strategia

**Cosa mostra:** Un confronto multidimensionale che permette di valutare a colpo d'occhio quale strategia ha il profilo migliore su tutte le metriche contemporaneamente.

**Come interpretarlo:**
- La strategia con l'area più grande e regolare è quella con le prestazioni più equilibrate.
- "Outside OK%" indica se l'area al di fuori della ROI è rimasta inalterata: 100% significa che l'embedding è stato confinato correttamente.

---

## Grafici Riassuntivi Cross-Test

Questi grafici combinano dati da più test per una visione d'insieme.

### `cross_boxplot_psnr_blocksize.png`

**Tipo:** Box & Whisker plot

**Asse X:** Block Size (4, 8, 16)

**Asse Y:** PSNR medio (dB)

**Cosa mostra:** La distribuzione del PSNR per ciascun block size, aggregando tutti i delta, sv_range e messaggi dal test sweep.

**Come interpretarlo:**
- La mediana (linea nel box) indica la tendenza centrale.
- Box stretti indicano un comportamento stabile; box larghi indicano alta variabilità.
- Outlier (punti fuori dai baffi) sono configurazioni anomale.

---

### `cross_boxplot_psnr_svrange.png`

**Tipo:** Box & Whisker plot

**Asse X:** SV Range (first, mid, last)

**Asse Y:** PSNR medio (dB)

**Cosa mostra:** Come la scelta del range di valori singolari influenza la distribuzione del PSNR.

**Come interpretarlo:**
- `mid` tipicamente offre il miglior compromesso: i valori singolari centrali contribuiscono meno all'energia dell'immagine, quindi modificarli causa meno distorsione visibile.
- `first` potrebbe causare più distorsione perché i valori singolari principali catturano la struttura dominante.

---

### `cross_correlation_matrix.png`

**Tipo:** Matrice di correlazione (Pearson)

**Variabili:** δ (delta), block_size, PSNR, SSIM, BER, Correct%

**Colore:** Scala blu-rosso (blu = correlazione negativa, rosso = correlazione positiva)

**Cosa mostra:** Le relazioni lineari tra parametri di configurazione e metriche di qualità/accuratezza.

**Come interpretarlo:**
- **δ vs PSNR:** Correlazione fortemente negativa attesa (più delta → meno qualità).
- **δ vs BER:** Correlazione negativa attesa (più delta → meno errori in condizioni ideali).
- **PSNR vs SSIM:** Correlazione positiva attesa (entrambe misurano la qualità).
- Correlazioni inattese possono rivelare interazioni non lineari tra i parametri.

---

## Come Generare i Grafici

```bash
# 1. Eseguire i test per generare i CSV
python tests/test_dataset.py

# 2. Generare tutti i grafici
python tests/plot_results.py
```

I grafici vengono salvati in formato PNG nella cartella `test_output/plots/`.

La costante `CSV_DIR` in cima a `tests/plot_results.py` definisce il percorso dei CSV di input (default: `test_output/dataset/`).
