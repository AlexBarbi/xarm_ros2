# Selezionatore pezzi: pick and place con xArm5

`pick_and_place.py` svuota il tavolo di `shared/world.usd`: trova i pezzi con la camera di profondità sul polso, riconosce il codice (part number) di ognuno confrontandolo con il CAD, sceglie dove prenderlo e lo lascia cadere nel suo contenitore. Ripete finché sul tavolo non resta niente.

Della posizione dei pezzi non c'è niente scritto nel codice. Sono fissi solo la posizione del tavolo e degli 11 contenitori (presi da `world.usd`) e la tabella `PIECE_BINS`, che dice in quale contenitore va ogni codice.

| File | Cosa contiene |
|---|---|
| `pick_and_place.py` | Nodo ROS 2: ciclo di lavoro, movimenti con MoveIt, pinza, controllo della presa |
| `piece_vision.py` | Percezione, senza ROS: mappa delle altezze, separazione dei pezzi, riconoscimento, scelta della presa |
| `shared/piece_models.json` | Libreria CAD dei pezzi (11 codici): punti della superficie, baricentro, volume |
| `shared/export_piece_models.py` | Rigenera la libreria dai file `.usd` dei pezzi (va eseguito nel container di Isaac Sim, serve `pxr`) |

## Come si lancia

Prerequisiti:

- **Isaac Sim** con `shared/world.usd` aperto e Play premuto. Il file ha i limiti dei giunti aperti al range reale del braccio: `joint1` e `joint5` a ±360°, `joint3` fino a −225°, `joint4` fino a +180°. La versione precedente è salvata in `shared/world.usd.before_joint_limits`.
- **MoveIt**: `ros2 launch xarm_moveit_config xarm5_moveit_isaac.launch.py`. Questo launch passa `limited:=false`, quindi MoveIt usa gli stessi limiti del file USD. Via SSH o senza display parte da solo senza RViz; si può forzare con `headless:=true` o `headless:=false`.

Poi, con `install/setup.bash` caricato:

```bash
cd ~/Downloads/ros_ws                                              # MODELS è un percorso relativo: shared/piece_models.json
python3 src/xarm_ros2/python/pick_and_place.py --dry-run           # guarda il tavolo e stampa cosa farebbe, senza prendere niente
python3 src/xarm_ros2/python/pick_and_place.py                     # smista tutto
```

Con il nastro, sempre da `ros_ws`:

```bash
python3 src/xarm_ros2/python/pick_and_place.py --conveyor      # giro dopo giro, finché non premi Ctrl-C
```

Il robot aspetta che il nastro si fermi, smista, e dice al nastro di ripartire. Si parlano con due file in `shared/`: `.conveyor_belt` lo scrive il nastro, `.conveyor_robot` lo scrive lo script. Se lo script non c'è, il nastro riparte quando non ha più pezzi sopra. In ogni caso riparte dopo `MAX_WAIT` (300 s simulati), così un robot bloccato non ferma la linea.

Alla fine stampa `done: N dropped into the bins, M on the table, K lost`. Esce con codice 1 se qualcosa è rimasto sul tavolo o è andato perso.

Il nodo usa solo `moveit_msgs` (non serve `moveit_py`):

- **Servizi:** `/compute_ik`, `/compute_fk`, `/compute_cartesian_path`, `/apply_planning_scene`, `/get_planning_scene`
- **Azioni:** `/move_action`, `/execute_trajectory`
- **Pinza e camera, trovate da sole all'avvio**, così lo stesso script va in Isaac e sul robot reale senza modifiche:
  - **pinza:** l'azione che finisce in `xarm_gripper/gripper_action` (driver reale, `GripperCommand`) o in `xarm_gripper_traj_controller/follow_joint_trajectory` (Isaac);
  - **camera:** il topic che finisce in `/aligned_depth_to_color/image_raw`, cioè `/camera/...` in Isaac e `/camera/camera/...` con realsense-ros 4 lanciato con `align_depth.enable:=true`, più il `color/camera_info` accanto;
  - il log dice cosa ha trovato: `gripper: …` e `camera: …`; `--depth-topic` sceglie un'altra immagine, per esempio quella con il rumore.

## Il ciclo di lavoro

1. **Scansione.** Il braccio va nella posa di scansione (`SCAN_XYZ`, pinza verso il basso) e scatta un'immagine di profondità.
2. **Riconoscimento.** `piece_vision` costruisce la mappa del tavolo, separa i pezzi e li riconosce. Quelli non riconosciuti vengono guardati più da vicino (vedi Percezione).
3. **Tavolo vuoto?** Se non c'è niente da prendere, il braccio aggiunge altre due viste del tavolo prima di crederci.
4. **Smistamento.** I pezzi trovati vengono presi uno dopo l'altro, **dal più alto al più basso**: quelli alti cadono se qualcosa li tocca.
5. **Nuova scansione.** Si torna al punto 1, al massimo 5 scansioni (`MAX_ROUNDS = 4` più quella finale).

Se un pezzo scivola durante il sollevamento, si riprova subito con un'altra presa nello stesso punto, al massimo 3 prese per pezzo. Se invece cade durante il trasporto, il pezzo può essere atterrato ovunque, anche sul prossimo da prendere, quindi si interrompe il giro e si riscansiona.

## Percezione (`piece_vision.py`)

### 1. Mappa delle altezze

Ogni pixel dell'immagine di profondità diventa un punto 3D nel frame `link_base`, con la posa della camera presa da `/compute_fk` e gli intrinseci da `camera_info`. I punti finiscono in una griglia di 1 mm sull'area di lavoro (`WORK_X`, `WORK_Y`), che tiene per ogni cella l'altezza massima sopra il tavolo e se la camera l'ha vista.

L'immagine viene presa solo dopo che il braccio si è fermato. Più viste si sommano nella stessa mappa.

### 2. Separazione dei pezzi

- **Pulizia:** una chiusura morfologica 3×3 riempie i buchi tra i pixel della camera. Il cerchio della base del robot (`ROBOT_KEEP_OUT`, 14 cm) viene ignorato.
- **Sopra i 4 mm è pezzo:** tutto quello che supera 4 mm d'altezza conta come pezzo.
- **Parti vicine:** un'apertura toglie i granelli, poi una dilatazione di 3 celle tiene unite le parti di uno stesso pezzo collegate da un collo sottile.
- **Ombre:** le zone non viste piccole (meno di 3000 celle) sono l'ombra di qualcosa di alto e uniscono le due parti ai lati. Un pezzo alto non viene spezzato in due dalla propria ombra.
- **Scarto:** le regioni sotto 150 celle (1,5 cm²) vengono scartate.

Per ogni pezzo si calcolano il rettangolo minimo che lo contiene (lunghezza, larghezza, angolo), l'altezza massima e un baricentro stimato pesando le celle con la loro altezza.

### 3. Riconoscimento dal CAD

Da `piece_models.json` la libreria calcola le **pose di appoggio stabili** di ogni codice: le facce dell'involucro convesso su cui il pezzo sta in piedi, cioè quelle con la proiezione del baricentro dentro la faccia. Per ognuna salva una firma (altezza, lunghezza, larghezza).

Un pezzo visto viene riconosciuto se la firma più vicina differisce al massimo di **9 mm** su ciascuna delle tre misure (`SIGNATURE_TOLERANCE`).

### 4. Pezzi che si toccano

Due pezzi a contatto formano un'unica macchia sconosciuta. `split` la taglia a quote crescenti, ogni 2 mm: a una certa quota il pezzo più alto si stacca dal resto. Se quella fetta corrisponde a un codice entro **5 mm** (`SPLIT_TOLERANCE`, più stretta perché un pezzo di un oggetto può assomigliare a un altro oggetto intero), diventa un pezzo a sé.

Il resto della macchia rimane sconosciuto: una volta tolto il pezzo alto, alla scansione successiva sarà da solo. Due pezzi della stessa altezza a contatto restano una macchia unica e vengono lasciati sul tavolo (e segnalati).

### 5. Altre viste

- **Guardare più da vicino:** se resta qualcosa di sconosciuto, il braccio porta la camera sopra quel pezzo, al massimo a 0,5 m dalla base, e aggiunge la vista alla mappa. Fino a 3 volte per scansione, e smette se non migliora. Serve per i pezzi tagliati dal bordo dell'immagine o nascosti da altri.
- **Guardarsi intorno:** la vista di scansione copre il 77% dell'area di lavoro. Prima di dichiarare il tavolo vuoto il braccio aggiunge due viste (`MORE_VIEWS`) e arriva al 93%.

### 6. Scelta della presa

La presa si cerca direttamente sulla mappa delle altezze, senza usare il modello CAD. Così funziona su qualsiasi faccia il pezzo appoggi. Per ogni orientamento della pinza (ogni 15°) e ogni posizione lungo il pezzo (ogni 2 mm), si prova a scendere con le punte delle dita da 5 a 32 mm sotto la cima del pezzo. Una presa è valida se:

- quello che sta tra le dita è largo tra 10 e 68 mm ed è tutto di questo pezzo (non di un vicino);
- ai due lati c'è posto per le dita: 14 mm di spessore più 8 mm di margine, con tutto quello che c'è lì almeno 3 mm sotto le punte;
- le dita toccano il pezzo su entrambi i lati per almeno 6 mm della loro larghezza (32 mm);
- le punte restano almeno 3 mm sopra il tavolo.

Tra le prese valide vince quella col punteggio più basso:

```
distanza dal baricentro [mm] + 40 / mm di contatto − 500 × profondità [m]
```

Vicino al baricentro e ben dentro il pezzo: le prese alte scivolano. Le prese già tentate su quel pezzo vengono escluse: si considerano uguali entro 12 mm e 20°.

## Movimento

### Pose e IK

L'xArm5 ha 5 assi e raggiunge solo pose con la pinza **esattamente verticale**: `top_down(yaw)`. Per questo ogni movimento parte da un'IK sulla posa esatta. La posa obiettivo di MoveIt, che perturba l'orientamento entro le tolleranze, non trova soluzioni.

`compute_ik` fa partire il risolutore con `joint1` già rivolto verso il punto, e scarta le soluzioni col braccio rivolto all'indietro o col gomito piegato al contrario (`joint3 < −π`), riprovando fino a 5 volte. Di ogni soluzione tiene la versione con `joint1` e `joint5` più vicini a dove sono adesso: un giro completo è la stessa posa. I limiti li legge da `/robot_description`.

### Il percorso più corto

Tavolo e contenitori stanno dietro la base del robot. Con i limiti a ±178° il braccio doveva girare tutto davanti: 4,4–5,5 rad di `joint1` a ogni trasporto. Con i limiti aperti passa dietro la base, e ogni trasporto costa 1,2–2,3 rad.

`move_to` prova prima la **linea retta del TCP** (`/compute_cartesian_path`) e usa il planner OMPL solo se la retta non è fattibile, per esempio se è bloccata o passa troppo vicino alla base. Tre accorgimenti rendono affidabile la retta:

1. **Partenza verticale.** Il braccio reale è inclinato di un capello rispetto alla verticale, e un braccio a 5 assi non può inclinarsi di lato: correggere quell'inclinazione mentre `joint1` ruota non ha IK, neanche a 0,001 rad. La retta si calcola quindi da una partenza con la pinza esattamente verticale, poi la traiettoria parte da dove il braccio è davvero.
2. **Polso uniforme.** I punti della retta (ogni 5 mm) hanno un'orientazione che segue la rotazione del braccio, così `joint5` gira in modo uniforme invece di oscillare avanti e indietro (prima arrivava a 1,3 rad di movimento inutile).
3. **Niente salti.** Un salto di oltre 0,5 rad su un giunto tra due punti è l'IK che ribalta il braccio: la retta viene scartata.

Per i movimenti corti vicino ai pezzi (`move_linear`: discesa, sollevamento) il ripiego non è il planner libero, ma un movimento in giunti solo se nessun giunto si sposta più di 0,5 rad. Altrimenti è un errore: un braccio che si ribalta vicino al tavolo ci sbatte contro.

Un arrivo segnalato come fallito dal controller (`CONTROL_FAILED`) viene accettato se tutti i giunti sono entro 0,03 rad dall'obiettivo: la forza della pinza simulata spinge il polso fin lì.

## Presa e trasporto (`pick_and_drop`)

1. **Orientamento.** La pinza è simmetrica: `yaw` e `yaw + 180°` sono la stessa presa. Si sceglie quella che fa girare meno il polso.
2. **Apertura.** La pinza si apre alla larghezza del pezzo più 2×8 mm mentre il braccio si muove.
3. **Sopra il pezzo.** Il TCP va 35 mm sopra la cima del pezzo. La scatola del pezzo viene tolta dalla scena di MoveIt, perché le dita ci devono entrare, e la pinza scende in linea retta.
4. **Chiusura** (`close_on`): vedi sotto.
5. **Sollevamento.** La scatola del pezzo viene attaccata al TCP nella scena di MoveIt, così il planner lo tiene lontano da tavolo, contenitori e altri pezzi. Il pezzo sale in verticale finché il fondo è 15 mm sopra il bordo dei contenitori, o sopra il pezzo più alto rimasto sul tavolo.
6. **Trasporto.** Il polso resta fermo e il pezzo ruota insieme al braccio: al contenitore non importa come cade. Il centro del pezzo va sopra il punto di rilascio, e lo scostamento tra TCP e centro del pezzo ruota anche lui.
   - **Pezzi bassi** (sotto i 70 mm del bordo): al centro del contenitore.
   - **Pezzi alti:** in un angolo, a 37 mm dal centro su x e y. Prima gli angoli lontani dal lato da cui arriva il braccio, così non ci passa sopra. Al centro cadrebbero e resterebbero appoggiati fuori dal bordo; in un angolo possono cadere solo verso l'interno.
7. **Rilascio.** Tutto si rilascia col fondo 15 mm sopra il bordo, anche i pezzi alti. Scendere di più porterebbe le dita sopra il bordo, e un dito che lo tocca mentre si apre fa esplodere la pinza simulata. Le dita si aprono di 0,3 rad e la scatola viene tolta dalla scena.

### La pinza (`close_on`)

Il driver dell'xArm reale non dà la forza della pinza, solo la posizione delle dita: tutto si basa su quella. Sul robot reale la pinza è l'azione `GripperCommand` `xarm_gripper/gripper_action`, in simulazione `FollowJointTrajectory` sul controller della pinza; lo script usa quella che trova.

- **Chiude oltre il pezzo:** le dita vengono mandate 8 mm più strette della larghezza vista, piano (0,35 rad/s), così si fermano sul pezzo e continuano a stringere con la forza a cui il motore è limitato.
- **Contatto:** le dita restano ferme per 0,2 s.
- **Mancato:** se le dita arrivano al punto in cui sono state mandate, tra loro non c'è niente.
- **Stringe:** l'obiettivo resta almeno 0,07 rad oltre il contatto misurato. Con meno margine le dita, che si flettono sotto la presa, arrivano così vicine all'obiettivo che un pezzo ancora in mano sembra perso.

I movimenti liberi delle dita sono limitati a 1 rad/s: aprire più in fretta subito dopo una presa strappa il meccanismo simulato.

### Controllo della presa durante il trasporto

Mentre c'è un pezzo in mano, ogni movimento del braccio guarda le dita ogni 5 ms. Finché restano lontane dall'obiettivo, il pezzo è tra loro. La posizione delle dita però non basta a dire che il pezzo è perso: con le posizioni vere dei pezzi in Isaac si vedono due falsi allarmi.

- **Le dita arrivano all'obiettivo con il pezzo in mano.** Un pezzo piatto si gira tra le dita, da disteso a di costa, e le dita si chiudono fino all'obiettivo mentre il pezzo sale con loro. Lo script allora comanda la chiusura completa; nel log compare `the fingers closed further, pressing to 0.85 rad`.
- **Le dita si chiudono del tutto con il pezzo in mano.** Succede quando tra i cuscinetti resta una parte sottile del pezzo: un 53152 è arrivato nel suo contenitore così.

Per questo una perdita si decide solo **sopra il tavolo**:

- **Fine della salita:** il braccio aspetta che le dita siano ferme. Se sono chiuse del tutto, il pezzo è scivolato: la pinza si apre lì, nel log compare `the piece slipped out: lost during lift, around (x, y, z)` e il tavolo viene riscansionato. Il pezzo ricade da fino a 10 cm a qualche centimetro di distanza, quindi le prese calcolate sulla vecchia mappa lo mancherebbero.
- **Sopra i contenitori le dita non si aprono mai prima del contenitore giusto.** Aprirle dove il braccio si trova farebbe cadere un pezzo ancora in mano nel contenitore sbagliato. Se le dita si chiudono del tutto per strada, il braccio va comunque al suo contenitore, rilascia e riscansiona il tavolo. Nel log compare `the fingers closed all the way during over bin N: the piece may have fallen on the way`.

## Scena di MoveIt

- **Fissi:** il tavolo (una scatola con la superficie 5 mm sotto il piano, così non tocca la base del robot né i pezzi) e gli 11 contenitori (fondo e quattro pareti).
- **A ogni giro:** i pezzi trovati come scatole con 1 cm di margine. Quella del pezzo da prendere viene tolta prima di scendere, attaccata alla pinza durante il trasporto e rimossa al rilascio. Durante discesa e salita si tolgono anche le scatole dei pezzi entro 3 cm: lo spazio per le dita lo garantisce già la mappa delle altezze, mentre con il margine il pezzo in mano risultava già in collisione con il vicino e MoveIt rifiutava la salita (errore -10). Tornano nella scena appena finita la salita.
- **All'avvio:** si tolgono i pezzi rimasti da un'esecuzione interrotta.

## Parametri principali

| Parametro | Valore | Cosa regola |
|---|---|---|
| `PIECE_BINS` | 11 codici → contenitori 0-10, 8 attivi | Quale codice va dove. È un'ipotesi: il foglio dice solo un codice per contenitore |
| `SCAN_XYZ`, `MORE_VIEWS` | 1 + 2 viste a 0,397 m | Copertura dell'area di lavoro (77%, poi 93%) |
| `FREE_SPEED`, `CARRY_SPEED`, `LINEAR_SPEED` | 1.0, 1.0, 1.0 | Frazione dei limiti di velocità dei giunti; `ACCELERATION` 0.8 |
| `LINE_STEP` | 5 mm | Passo dei punti della linea retta |
| `CARRY_CLEARANCE` | 35 mm | Fondo del pezzo sopra bordi e pezzi durante il trasporto |
| `BIN_CORNER` | 37 mm | Scostamento dal centro per i pezzi alti |
| `GRIPPER_CLOSE_SPEED` / `GRIPPER_SPEED` | 0,35 / 1,0 rad/s | Chiusura sul pezzo / movimenti liberi delle dita |
| `GRIPPER_FORCE` | 20 N | Forza sul pezzo: `max_effort` della pinza reale (il driver di questo workspace lo ignora), 2,2 N·m in `world.usd` |
| `GRIPPER_MISS_MARGIN` / `GRIPPER_PRESS` | 8 mm / 0,07 rad | Quanto oltre il pezzo si mandano le dita / margine minimo oltre il contatto |
| `GRIPPER_HELD_ERROR` / `GRIPPER_FREE_SAMPLES` | 0,01 rad / 3 campioni | Dita "all'obiettivo", cioè niente tra loro |
| `SIGNATURE_TOLERANCE` / `SPLIT_TOLERANCE` | 9 / 5 mm | Riconoscimento di un pezzo intero / di una fetta |
| `GRASP_ATTEMPTS`, `MAX_ROUNDS` | 3, 4 | Prese per pezzo, giri di scansione |

## Stato della validazione

Prove fatte in Isaac Sim senza interfaccia, con la posizione vera di ogni pezzo letta dalla fisica e un giudice indipendente che controlla se il pezzo è davvero dentro il contenitore giusto.

**Prova del 3 novembre**, 8 codici attivi, rumore della camera a 1, `CARRY_SPEED` e `LINEAR_SPEED` a 1.0:

- **Batch nuovo** (`batch_results/fresh1`, 8 corse da 5 pezzi): 30/40 nel contenitore giusto (75%). Per scenario:
  - table, tipped e touching: 9/10 ciascuno;
  - belt: 3/10, di cui una corsa esplosa (0/5).

  Senza la corsa esplosa sono 30/35. Gli errori:
  - 47531 in piedi preso per 51189;
  - un pezzo mai visto alla prima scansione;
  - un 57388 in piedi sul bordo del tavolo senza presa possibile;
  - sul nastro, pezzi lasciati in fondo dove la camera arriva male.
- **Nastro con `--conveyor`** (3 giri da 5 pezzi): lo scambio di stato tra nastro e robot funziona e i giri vanno avanti da soli. Due prove si sono fermate per un'esplosione della pinza simulata.
- **Esplosioni:** 5 nella giornata, tutte su prese strette (contatto a 0,71–0,77 rad, pezzi larghi 14–18 mm), durante la chiusura o la salita. Molte prese uguali vanno bene. È la fisica del meccanismo simulato, non il programma; il batch le riconosce (`Simulazione esplosa`) e ferma la corsa.

Prima del 3 novembre, con 3 codici e senza rumore: 10/10 sulla tua disposizione e 105/110 su disposizioni casuali.

## Limiti noti

- **Solo simulazione.** La pinza usa solo la posizione delle dita, come sul robot reale, ma l'azione `GripperCommand` del driver vero è stata provata solo con un sostituto (`batch/driver_gripper.py`). Il rumore della camera (`depth_noise.py`) imita una RealSense: con una camera vera le soglie vanno ritarate.
- **Copertura.** Il 7% dell'area di lavoro (gli angoli) non viene mai visto.
- **Pezzi a contatto della stessa altezza** restano una macchia unica e vengono lasciati sul tavolo, segnalati.
- **Rimbalzi.** Circa il 2% dei pezzi rimbalza fuori da un contenitore già pieno. Viene ripreso se è raggiungibile, altrimenti segnalato.
- **Barrette in piedi.** Restano appoggiate negli angoli, con la cima sopra il bordo fino a circa 8 mm oltre la parete interna.
- **47531 in piedi su uno spigolo** non ha una posa stabile nel CAD e viene preso per un 51189 in piedi (43×14×64 contro 42×20×68 mm).
- **Pinza simulata.** Su prese strette il meccanismo a catena chiusa a volte si strappa e lancia il pezzo: 5 volte in una giornata con il motore a 5 N·m (45 N). Ora `world.usd` ha 2,2 N·m, cioè 20 N sul pezzo (un motore per due dita, leva 0,055 m): nessuna esplosione in 8 corse sulle stesse disposizioni, troppo poche per dirlo risolto. Per un'altra forza: `batch_test.py --grip-torque` per provarla, `shared/set_grip_torque.py` per salvarla in `world.usd`.
- **Fondo del nastro.** Il quinto pezzo di un giro si ferma a x ≈ −0,62, dove la camera lo vede solo in parte: viene lasciato, segnalato, e preso al giro dopo.
- **Mappatura.** In `PIECE_BINS` ci sono tutti gli 11 codici, 3 commentati (47527, 52737, 56753). L'assegnazione ai contenitori è da confermare. 47527 e 56753 sono quasi gemelli e si possono confondere.
- **Posizioni fisse.** Tavolo e contenitori sono costanti prese da `world.usd`: se si spostano nella scena, vanno aggiornate nel codice.
- **Percorso del CAD.** `MODELS` è relativo alla cartella da cui si lancia lo script (va lanciato da `ros_ws`).
- **Sessione interattiva.** Lo script non è mai stato eseguito nella sessione di Isaac con interfaccia in streaming, che lì girava a circa 0,2 volte il tempo reale.
