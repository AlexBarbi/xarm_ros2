# shared

Quello che leggono Isaac Sim (montato in `/workspace` nel container) e `python/pick_and_place.py`:

- `world.usd`: scena con braccio, tavolo, nastro centrato e 6 casse (il programma le trova da solo all'avvio). Il robot è in `xarm5/`.
- `world_xarm6.usd`: la stessa scena con l'xArm6 degli asset di Isaac Sim, che Isaac scarica dalla rete aprendola. Si rifà da `world.usd` con `build_world_xarm6.py` (comando in `python/COMANDI.md`, punto 5).
- `conveyor_cycle.py`: lo script del nastro, attaccato a `/World/Conveyor`.
- `piece_models.json`: le forme dei pezzi per il riconoscimento, generate con `export_piece_models.py`.
- `set_grip_torque.py`: cambia la forza della pinza in `world.usd`. Ora è 2,2 N·m, cioè 20 N sul pezzo.
- `mold.txt`: lo stampo in produzione, letto dal nastro e dal robot.
- `fastdds_udp.xml`: configurazione DDS per parlare con il container.

Non sono nel repo, vanno copiati a mano:

- `Studio selezionatore oggetti Volpiano/`: i CAD dei pezzi. Servono al nastro e ai batch, che ci prendono i pezzi da far cadere.
- `synthetic_*.pt`: il modello YOLO, che il programma non usa.

## Su un altro PC

Il codice cerca `shared/` dentro `ros_ws`. Dopo aver clonato il repo in `ros_ws/src/xarm_ros2`:

```bash
cd ~/Downloads/ros_ws
ln -s src/xarm_ros2/shared shared          # lo stesso link di questo PC
setfacl -m u:1234:rwx,d:u:1234:rwx,d:u:$USER:rwx src/xarm_ros2/shared   # Isaac nel container (utente 1234) ci scrive
setfacl -m u:1234:rw src/xarm_ros2/shared/world.usd
cp -r "/percorso/Studio selezionatore oggetti Volpiano" shared/
```

Poi il container di Isaac Sim va lanciato con `-v ~/Downloads/ros_ws/shared:/workspace`.
