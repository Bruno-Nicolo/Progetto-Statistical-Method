# ROADMAP PROGETTO SVD e YOLO: Image Steganography

Questo documento descrive la pipeline di sviluppo per implementare un sistema di steganografia semantica. L'obiettivo è nascondere informazioni all'interno di un'immagine utilizzando YOLO per l'individuazione della Region of Interest (ROI) e un algoritmo SVD scritto da zero in Python per l'embedding matematico.

---

## Fase 1: Implementazione Matematica della SVD "from scratch" in Python

In questa fase si costruisce il motore matematico. [cite_start]Partendo da una matrice dati X, la SVD fattorizza la matrice in X = U _ Sigma _ V^T[cite: 276, 400]. [cite_start]Le matrici U e V hanno colonne ortonormali, mentre Sigma è una matrice diagonale con i valori singolari ordinati in modo decrescente[cite: 277].

1. **Gestione del DATO base**: Convertire l'immagine (o la ROI estratta) in una matrice numpy bidimensionale X.
   - **SCELTA DA FARE: Mean Centering**. Decidere se sottrarre la media da ogni feature dell'immagine. [cite_start]Il mean centering assicura di catturare la struttura intrinseca dei dati [cite: 121] [cite_start]ed è essenziale per la PCA tradizionale[cite: 46], ma nella compressione/steganografia standard di immagini grezze a volte viene omesso. Devi testare quale approccio preserva meglio l'immagine ricostruita.
2. [cite_start]**Calcolo della matrice V**: Calcolare la matrice di covarianza C, proporzionale a X^T \* X[cite: 266, 279]. Per trovare gli autovalori e autovettori di C, si può utilizzare il metodo delle potenze. Le colonne di V saranno gli autovettori (direzioni principali)
3. [cite_start]**Calcolo della matrice U**: Lavorare con la matrice di Gram S, proporzionale a X \* X^T[cite: 326, 327]. [cite_start]La scomposizione agli autovalori di S restituisce la matrice U[cite: 336, 338].
4. [cite_start]**Calcolo di Sigma**: I valori singolari in Sigma si ottengono dalle radici quadrate degli autovalori[cite: 287]. [cite_start]Gli autovalori non nulli calcolati da X^T _ X e X _ X^T coincidono[cite: 341].
5. **Validazione**: Confrontare i risultati della SVD custom con `numpy.linalg.svd` usando matrici di test di piccole dimensioni.

---

## Fase 2: Selezione della ROI con YOLOv8

Integrazione del modello di Deep Learning per rendere la steganografia contestuale.

1. **Setup YOLO**: Caricare un modello pre-addestrato (es. YOLOv8n) tramite la libreria `ultralytics`.
2. **Inferenza**: Passare l'immagine di "cover" al modello per ottenere le coordinate (bounding boxes) degli oggetti presenti.
3. **Strategia di Embedding (Sfondo vs Soggetto)**. Far decidere all'utente dove nascondere il messaggio, permetti di scegliere tra:

- _Opzione A_: Nascondere nei bounding box degli oggetti (texture complesse coprono meglio le alterazioni).
- _Opzione B_: Nascondere nello sfondo (escludendo i bounding box) per non alterare i soggetti principali.
- _Opzione C_: Scegliere dinamicamente il bounding box più grande.

---

## Fase 3: Embedding del Messaggio

Fusione dei dati all'interno dei valori singolari.

1. **Preparazione del payload**: Convertire il messaggio segreto in una sequenza binaria usando codifica ASCII a 7 bit (128 caratteri), riducendo la probabilità di errore rispetto a UTF-8.
2. **Suddivisione in blocchi**: Dividere la ROI scelta in blocchi più piccoli (es. 8x8 o 16x16 pixel) per ottimizzare la velocità dell'SVD custom.
3. **Modifica di Sigma**: Per ogni blocco, applicare la SVD custom per ottenere U, Sigma, V^T. Alterare i valori di Sigma aggiungendo il payload.

- [cite_start]**Quali Valori Singolari Alterare?** Il primo valore singolare cattura la varianza massima[cite: 243, 363]; modificarlo creerà artefatti visibili. Modificare gli ultimi valori è invisibile ma vulnerabile alla compressione JPEG. Modificare i valori intermedi è il miglior compromesso. Nel main del programma aggiungi un flag per permettermi di fare dei test e valutare l'opzione migliore.

4. [cite_start]**Ricostruzione dell'Immagine**: Ricalcolare i blocchi modificati moltiplicando U _ Sigma_modificata _ V^T[cite: 276]. Inserire i blocchi al loro posto originario e salvare l'immagine (Stego-Image).

---

## Fase 4: Estrazione del Messaggio

Il processo inverso per recuperare i dati.

1. **Rilevamento YOLO**: L'immagine ricevuta viene ripassata sotto YOLO per ritrovare le coordinate esatte della ROI utilizzata in fase di embedding.
2. **Decomposizione SVD**: Applicare nuovamente l'SVD ai blocchi della ROI estratta per ottenere la nuova matrice Sigma_modificata.
3. **Estrazione Informed**:
   - L'estrazione richiede l'immagine originale come riferimento. I bit vengono recuperati confrontando i valori singolari della stego-image con quelli dell'originale, rendendo la decodifica molto più affidabile rispetto all'estrazione blind.

---

## Fase 5: Valutazione delle Performance

1. **Metriche Visive**: Misurare il Peak Signal-to-Noise Ratio (PSNR) e lo Structural Similarity Index (SSIM) tra l'immagine originale e la stego-image per provare l'impercettibilità.
2. **Metriche di Robustezza**: Provare ad applicare piccoli filtri, rumore o compressione all'immagine e verificare se il payload è ancora estraibile e se YOLO riconosce ancora i bounding box.
