# Comandi

Come avviare Isaac Sim, MoveIt e lo smistatore, e come lanciare il batch. Cosa fa il programma e come funziona è spiegato in [pick_and_place.md](pick_and_place.md).

Tutti i percorsi partono da `~/Downloads/ros_ws`. Sull'host `/opt/ros/jazzy/setup.bash` è già in `~/.bashrc`.

## 0. Una volta sola, su un PC nuovo

```bash
cd ~/Downloads/ros_ws/src
git clone --recursive -b jazzy https://github.com/AlexBarbi/xarm_ros2.git
cd ~/Downloads/ros_ws
ln -s src/xarm_ros2/shared shared                                    # il codice e il container cercano ros_ws/shared
cp -r "/percorso/Studio selezionatore oggetti Volpiano" shared/       # i CAD dei pezzi non sono nel repo
rosdep install --from-paths src --ignore-src --rosdistro jazzy -y
colcon build
```

Il container di Isaac Sim va creato una volta sola:

```bash
docker run --name isaac-sim --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -v ~/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
    -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
    -v ~/.cache/ov/hub:/var/cache/hub:rw \
    -v ~/Downloads/ros_ws/shared:/workspace:rw \
    -u 1234:1234 \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/fastdds_udp.xml \
    nvcr.io/nvidia/isaac-sim:6.2.0
```

`~/docker/isaac-sim/...` deve esistere ed essere dell'utente 1234 (`sudo mkdir -p` e `sudo chown -R 1234:1234`).

## 1. Isaac Sim (terminale 1)

```bash
docker start -ai isaac-sim                 # se il container esiste ma è fermo
# docker exec -it isaac-sim bash           # se è già acceso
./isaac-sim.streaming.sh --/app/scripting/ignoreWarningDialog=true
```

- Senza quell'opzione lo script del nastro chiede il permesso in una finestra, e finché non lo dai il nastro non parte.
- Per cambiare la pausa del nastro: `CONVEYOR_WAIT=60 ./isaac-sim.streaming.sh --/app/scripting/ignoreWarningDialog=true`. Il default è 3 s; il nastro aspetta comunque che il robot abbia finito.

Collegati con l'**Isaac Sim WebRTC Streaming Client** all'IP del PC, apri `File > Open > /workspace/world.usd` e premi **Play**.

## 2. MoveIt (terminale 2)

```bash
source ~/Downloads/ros_ws/install/setup.bash
ros2 launch xarm_moveit_config xarm5_moveit_isaac.launch.py
```

Via SSH parte da solo senza RViz. `headless:=false` forza RViz se hai un display.

## 3. Smistatore (terminale 3)

Va lanciato da `ros_ws`, perché `shared/piece_models.json` è un percorso relativo.

```bash
cd ~/Downloads/ros_ws
source install/setup.bash
python3 src/xarm_ros2/python/pick_and_place.py --dry-run      # guarda e dice cosa farebbe, senza muovere la pinza
python3 src/xarm_ros2/python/pick_and_place.py                # smista quello che c'è sul tavolo ed esce
python3 src/xarm_ros2/python/pick_and_place.py --conveyor     # segue il nastro giro dopo giro, fino a Ctrl-C
```

All'avvio scrive cosa ha trovato: `gripper: …` e `camera: …`.

Con il rumore della camera, in un quarto terminale:

```bash
cd ~/Downloads/ros_ws && source install/setup.bash
python3 src/xarm_ros2/python/depth_noise.py --strength 1
```

Poi allo smistatore aggiungi `--depth-topic /camera/aligned_depth_to_color/image_noisy`.

Per fermare tutto: Ctrl-C sullo smistatore, poi su MoveIt, poi Stop in Isaac.

## 4. Batch

Il batch fa tutto da solo: Isaac Sim senza interfaccia dentro il container `isaac-sim`, MoveIt, rumore e smistatore, una corsa dopo l'altra. Usa `ROS_DOMAIN_ID` 78, così non disturba il resto.

Prima di lanciarlo:

- il container deve essere acceso: `docker start isaac-sim`;
- **non** servono i terminali 1–3. Chiudi l'Isaac con interfaccia, perché due Isaac sulla stessa GPU rallentano tutto.

```bash
cd ~/Downloads/ros_ws
python3 src/xarm_ros2/python/batch/batch_test.py --smoke              # prova: 4 corse, una per scenario, circa 10 minuti
```

Per una sessione lunga via SSH lancialo con `nohup`, così continua anche se ti scolleghi:

```bash
nohup python3 src/xarm_ros2/python/batch/batch_test.py --hours 12 > batch.out 2>&1 &
tail -f batch.out                                                      # una riga per corsa
```

Per fermarlo: `kill <pid>` oppure Ctrl-C se è in primo piano. I risultati raccolti fin lì restano.

| Cosa | Comando |
|---|---|
| Per un numero di corse | `--runs 40` |
| Solo alcuni scenari | `--scenarios belt,touching` (`table`, `tipped`, `touching`, `belt`) |
| Rumore della camera | `--noise 1` (default `0,1`: metà corse senza, metà con) |
| L'azione della pinza reale (sostituto) invece del controller | `--gripper driver` oppure `both` |
| Un'altra forza della pinza | `--grip-torque 3.3` (N·m = N sul pezzo × 0,11; `world.usd` ha 2,2, cioè 20 N) |
| Riprendere una sessione interrotta | `--out batch_results/<cartella>` (la stessa) |
| Rifare corse precise | `--replay table:7590197 belt:48460314` (scenario:seed, li trovi nel report) |
| Rifare il report dai dati | `--report batch_results/<cartella>` |

I risultati vanno in `batch_results/<data_ora>/` e si aggiornano dopo ogni corsa:

- `summary.md`: il report. Contiene:
  - la percentuale di pezzi nel bin giusto, con l'intervallo di confidenza;
  - i risultati per scenario e per codice;
  - gli scambi di riconoscimento;
  - le corse andate male, con il loro log.
- `runs.csv`, `pieces.csv`, `events.csv`: i dati, una riga per corsa, per pezzo e per evento.
- `logs/`: i log delle corse fallite e delle prime riuscite.

Le corse con stato **Simulazione esplosa** sono quelle in cui il meccanismo della pinza simulata si è strappato. È un problema di Isaac, non del programma: valutale a parte.

## Forza della pinza in world.usd

```bash
docker exec isaac-sim bash -c "cd /tmp && /isaac-sim/python.sh /workspace/set_grip_torque.py 2.2"   # N·m, con world.usd chiuso
```
