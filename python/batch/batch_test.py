#!/usr/bin/env python3
"""
Long test of the sorter: runs pick_and_place.py on layout after layout in headless Isaac Sim, and counts what really
happened from the true positions of the pieces, not from what the script says about itself.

  python3 batch_test.py --smoke                 # first of all: one run of each scenario, a few minutes
  python3 batch_test.py --hours 12              # as long as you want, Ctrl-C stops it and keeps everything
  python3 batch_test.py --report batch_results/20261102_1530    # the report again, from the data
  python3 batch_test.py --hours 2 --grip-torque 3.3 --out batch_results/torque_3.3   # another gripper force (30 N), to compare

Results go in batch_results/<date>/, written after every run:
  summary.md    the report: rates overall, by scenario and for each code, what got confused with what, the failed runs
  pieces.csv    one row for each piece of each run
  runs.csv      one row for each run
  events.csv    what went wrong for each code (slips, empty grasps, losses)
  logs/         the log of the runs that failed, and of the first few

Scenarios: table (pieces at random, on the table), tipped (the same, many on their side), touching (pairs of pieces in
contact), belt (a round of the conveyor), scene (the pieces saved in world.usd, if any). --out on an existing folder
goes on from there.

"Simulazione esplosa" in the report: the simulated gripper linkage tore (the arm half a radian or more away from where it
is told to be, for half a second), and the run stopped there. It happens on narrow grasps, during the close or the lift.
It is Isaac Sim, not the sorter: count those runs apart. With 5 N*m (45 N) there were 5 in a day of tests, with 2.2 N*m
(20 N, now in world.usd) none in 8 runs on the same layouts: too few to call it fixed.

Runs one sort at a time, without --conveyor: on the belt it starts when the belt has stopped, and the belt waits for it.
Each run takes the parts of one mold, round and round the molds (--molds, all of shared/piece_models.json by default),
on the belt and on the table alike: the sorter is told the mold in a mold.txt of its own, and so is the belt.

Needs Isaac Sim in a container (--container) or installed here (--container '' --isaac-path), the workspace built, and
ROS 2 sourced by --ros-setup. Everything else is found from where this file is. Nothing is saved in world.usd.
"""

import argparse
import ast
import collections
import csv
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from piece_vision import mold_bins, mold_parts  # noqa: E402
SCENARIOS = ('scene', 'table', 'tipped', 'touching', 'belt')
RUN_FIELDS = ['run', 'scenario', 'noise', 'gripper', 'seed', 'mold', 'status', 'exit', 'pieces', 'ok', 'wrong_bin', 'on_table',
              'lost', 'robot_s', 'wall_s', 'real_time_factor', 'in_window', 'drops', 'silent_failure', 'false_alarm',
              'straight_lines', 'around', 'moveit_errors']
PIECE_FIELDS = ['run', 'scenario', 'noise', 'gripper', 'seed', 'mold', 'piece', 'code', 'target_bin', 'outcome', 'where', 'said',
                'recognition', 'seen_mm', 'grasp_found', 'x', 'y', 'over_rim_mm']
EVENT_FIELDS = ['run', 'scenario', 'noise', 'code', 'kind', 'count']
MAX_JOINT4 = 3.2                # rad: past its stop it has torn loose, even if the fingers are still somewhere near
MIN_FINGER = -0.1               # rad of drive_joint: the same, it stops at 0
MAX_TRACKING = 0.5              # rad between where an arm joint is told to be and where it is, for half a second: the
                                # gripper linkage tore and the arm with it (normal motions at full speed: 0.11)
BIN_MARGIN = 0.001              # m past the inner wall of a bin that still counts as inside
MATCH_DISTANCE = 0.07           # m between where the camera says a piece is and where it is
LOG_LINE = re.compile(r'^\[\w+\] \[[\d.]+\] \[pick_and_place\]: (.*)$')
SEEN = re.compile(r'^(?:(\d{12})|unknown piece), (\d+) x (\d+) x (\d+) mm at \((-?[\d.]+), (-?[\d.]+)\)(.*)$')
CODE_EVENT = re.compile(r'^(\d{12}): (.*)$')
PLACED = re.compile(r'^(\d{12}) placed,')
DONE = re.compile(r'^done: (\d+) dropped into the bins, (?:\d+ into the reject bin, )?(\d+) on the table, (\d+) lost, ([\d.]+) s')
EVENT_KINDS = (('slipped out', 'slip'), ('may have fallen', 'closed_in_transport'), ('lost during', 'lost_in_transport'),
               ('nothing between', 'empty_grasp'),
               ('MoveIt error', 'moveit_error'), ('no collision-free IK', 'no_ik'), ('folded', 'no_ik'))


STARTED = set()                 # still to be stopped: a stop can come while the others are being stopped, and skip them


class SimDown(Exception):
    """Isaac Sim hung or died: not a result of the run, it goes again."""


class StackDown(Exception):
    """MoveIt did not come up."""


# ---------------------------------------------------------------- reading what happened

def read_worst(path):
    """What watch_joints.py has seen so far, {} before it has written anything."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def sorter_constants(path):
    """The bins, the table and the work area of pick_and_place.py, without importing it: it needs ROS."""
    found = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') in (
                'BINS_XY', 'BIN_INNER', 'BIN_HEIGHT', 'TABLE_TOP_Z', 'REJECT_BIN', 'WORK_X', 'WORK_Y'):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    return found


def parse_log(text):
    """What the sorter said: the pieces of its first look at the table, and how each thing went."""
    seen, events = [], collections.Counter()
    counts = collections.Counter()
    drops = collections.Counter()
    done = None
    first_look = True
    for line in text.splitlines():
        match = LOG_LINE.match(line)
        if not match:
            continue
        message = match.group(1)
        if message.startswith('above piece'):
            first_look = False
        if ': straight line to' in message:
            counts['straight_lines'] += 1
        if 'going around' in message:
            counts['around'] += 1
        if 'MoveIt error' in message:
            counts['moveit_errors'] += 1
        found = SEEN.match(message)
        if found:
            if first_look:
                seen.append({'said': found.group(1), 'size': [int(found.group(i)) for i in (2, 3, 4)],
                             'xy': (float(found.group(5)), float(found.group(6))), 'grasp': 'grasp (' in found.group(7)})
            continue
        placed = PLACED.match(message)
        if placed:
            drops[placed.group(1)] += 1
            continue
        finished = DONE.match(message)
        if finished:
            done = [int(finished.group(1)), int(finished.group(2)), int(finished.group(3)), float(finished.group(4))]
            continue
        coded = CODE_EVENT.match(message)
        if coded:
            kind = next((k for text_, k in EVENT_KINDS if text_ in coded.group(2)), 'other')
            events[(coded.group(1), kind)] += 1
    return {'seen': seen, 'events': events, 'counts': counts, 'drops': drops, 'done': done}


def where_is(piece, bins, inner):
    """(where it is, how far over the walls it sticks out above the rim [mm]). It is in a bin if the walls hold it."""
    low, high = piece['low'], piece['high']
    if high[2] < -0.2:
        return 'floor', 0.0
    mean = piece['mean']
    below_low, below_high = piece['below_low'], piece['below_high']
    mx, my = inner[0] / 2 + BIN_MARGIN, inner[1] / 2 + BIN_MARGIN
    for i, (x, y) in enumerate(bins):
        if below_low is None or low[2] < -0.003:
            break
        inside = (below_low[0] > x - mx and below_high[0] < x + mx and below_low[1] > y - my and below_high[1] < y + my and
                  abs(mean[0] - x) < mx and abs(mean[1] - y) < my)
        if inside:
            over = max(max(abs(low[0] - x), abs(high[0] - x)) - inner[0] / 2, max(abs(low[1] - y), abs(high[1] - y)) - inner[1] / 2)
            return 'bin {}'.format(i), max(over * 1000, 0.0)
    if -0.74 < mean[0] < 0.06 and -0.95 < mean[1] < 0.55 and -0.02 < low[2] < 0.05:
        return 'table', 0.0
    return 'elsewhere', 0.0


def judge(start, end, log, bins, inner, piece_bins):
    """One row for each piece: where it ended up against where it should, and whether it was recognized."""
    rows = []
    truth = []
    for name, piece in start['pieces'].items():
        center = ((piece['low'][0] + piece['high'][0]) / 2, (piece['low'][1] + piece['high'][1]) / 2)
        truth.append({'name': name, 'code': piece['number'], 'center': center, 'seen': None})
    for seen in log['seen']:                # a later look at the same place replaces an earlier one
        distances = [math.dist(t['center'], seen['xy']) for t in truth]
        if distances and min(distances) < MATCH_DISTANCE:
            truth[distances.index(min(distances))]['seen'] = seen
    for t in truth:
        piece = end['pieces'].get(t['name'])
        target = piece_bins.get(t['code'], -1)
        where, over = where_is(piece, bins, inner) if piece else ('gone', 0.0)
        if where == 'bin {}'.format(target):
            outcome = 'ok'
        elif where.startswith('bin'):
            outcome = 'wrong_bin'
        elif where == 'table':
            outcome = 'on_table'
        else:
            outcome = 'lost'
        seen = t['seen']
        if seen is None:
            recognition = 'not_seen'
        elif seen['said'] is None:
            recognition = 'unknown'
        else:
            recognition = 'right' if seen['said'] == t['code'] else 'wrong'
        rows.append({'piece': t['name'], 'code': t['code'], 'target_bin': target, 'outcome': outcome, 'where': where,
                     'said': (seen['said'] or '') if seen else '', 'recognition': recognition,
                     'seen_mm': 'x'.join(str(v) for v in seen['size']) if seen else '',
                     'grasp_found': int(bool(seen and seen['grasp'])), 'x': round(t['center'][0], 3),
                     'y': round(t['center'][1], 3), 'over_rim_mm': round(over, 1)})
    return rows


def wilson(hits, total):
    """95% interval of a rate: with few trials, 5 out of 5 does not mean 100%."""
    if not total:
        return 0.0, 0.0
    z = 1.96
    p = hits / total
    middle = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(middle - half, 0.0), min(middle + half, 1.0)


def rate(hits, total):
    if not total:
        return '-'
    low, high = wilson(hits, total)
    return '{:.1f}% ({:.0f}-{:.0f})'.format(100 * hits / total, 100 * low, 100 * high)


def percent(hits, total):
    return '{:.1f}%'.format(100 * hits / total) if total else '-'


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def table(head, rows):
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return lines + ['']


def write_report(folder, command=''):
    """summary.md from the csv files: it can be run again at any time, also while the batch goes on."""
    folder = Path(folder)
    runs = read_rows(folder / 'runs.csv')
    pieces = read_rows(folder / 'pieces.csv')
    events = read_rows(folder / 'events.csv')
    judged = [r for r in runs if r['status'] != 'infrastructure']
    out = ['# Batch del selezionatore', '']
    if command:
        out += ['Comando: `{}`'.format(command), '']
    ok = sum(p['outcome'] == 'ok' for p in pieces)
    clean = sum(r['ok'] == r['pieces'] and r['pieces'] != '0' for r in judged)
    robot = sum(float(r['robot_s']) for r in judged if r['robot_s'])
    wall = sum(float(r['wall_s']) for r in judged if r['wall_s'])
    reject = sorter_constants(HERE.parent / 'pick_and_place.py')['REJECT_BIN']
    rejected = sum(p['where'] == 'bin {}'.format(reject) for p in pieces)
    out += ['## In sintesi', '',
            '- **Pezzi nel contenitore giusto: {}**, {} su {}.'.format(rate(ok, len(pieces)), ok, len(pieces)),
            '- Nel bin degli scarti ({}): {}. Il batch usa solo codici noti, quindi ognuno è un pezzo non riconosciuto, '
            'contato tra i bin sbagliati.'.format(reject, rejected),
            '- Corse perfette: {}, {} su {}.'.format(rate(clean, len(judged)), clean, len(judged)),
            '- Tempo robot per pezzo: {:.1f} s simulati. Corse: {:.1f} ore reali in tutto.'.format(
                robot / len(pieces) if pieces else 0.0, wall / 3600),
            '- Tra parentesi l\'intervallo di confidenza al 95%.', '']
    silent = sum(r['silent_failure'] == '1' for r in judged)
    alarms = sum(r['false_alarm'] == '1' for r in judged)
    statuses = collections.Counter(r['status'] for r in runs)
    out += ['## Affidabilità', '']
    out += table(['Cosa', 'Corse', 'Quota'], [
        ['Fallimenti silenziosi: lo script dice ok ma un pezzo non è a posto', silent, percent(silent, len(judged))],
        ['Falsi allarmi: lo script dice errore ma è tutto a posto', alarms, percent(alarms, len(judged))],
        ['Simulazione esplosa', statuses['exploded'], percent(statuses['exploded'], len(judged))],
        ['Script oltre il tempo massimo', statuses['timeout'], percent(statuses['timeout'], len(judged))],
        ['Problemi di Isaac o MoveIt, corse non conteggiate', statuses['infrastructure'], percent(statuses['infrastructure'], len(runs))]])
    groups = collections.defaultdict(list)
    for r in judged:
        groups[(r['scenario'], r['noise'], r['gripper'])].append(r)
    rows = []
    for (scenario, noise, gripper), group in sorted(groups.items()):
        names = set(r['run'] for r in group)
        part = [p for p in pieces if p['run'] in names]
        hits = sum(p['outcome'] == 'ok' for p in part)
        seconds = sum(float(r['robot_s']) for r in group if r['robot_s'])
        windows = [r['in_window'] for r in group if r['in_window'] != '']
        rows.append([scenario, noise, gripper, len(group), len(part), rate(hits, len(part)),
                     percent(sum(r['ok'] == r['pieces'] for r in group), len(group)),
                     percent(sum(p['recognition'] == 'right' for p in part), len(part)),
                     '{:.1f}'.format(seconds / len(part)) if part else '-',
                     percent(sum(w == '1' for w in windows), len(windows))])
    out += ['## Per scenario', '']
    out += table(['Scenario', 'Rumore', 'Pinza', 'Corse', 'Pezzi', 'Nel bin giusto', 'Corse perfette', 'Riconosciuti',
                  's robot/pezzo', 'Entro la finestra'], rows)
    by_code = collections.defaultdict(list)
    for p in pieces:
        by_code[p['code']].append(p)
    count = collections.Counter()
    for e in events:
        count[(e['code'], e['kind'])] += int(e['count'])
    rows = []
    for code, part in sorted(by_code.items()):
        n = len(part)
        tally = collections.Counter(p['outcome'] for p in part)
        seen = collections.Counter(p['recognition'] for p in part)
        rows.append([code, part[0]['target_bin'], n, rate(tally['ok'], n), tally['wrong_bin'], tally['on_table'], tally['lost'],
                     percent(seen['right'], n), percent(seen['wrong'], n), percent(seen['unknown'], n), percent(seen['not_seen'], n),
                     percent(sum(p['recognition'] == 'right' and p['grasp_found'] == '0' for p in part), n),
                     count[(code, 'slip')], count[(code, 'empty_grasp')],
                     count[(code, 'closed_in_transport')] + count[(code, 'lost_in_transport')]])
    out += ['## Per codice', '',
            'Le ultime tre colonne contano gli eventi, attribuiti al codice che lo script credeva di avere in mano.', '']
    out += table(['Codice', 'Bin', 'Pezzi', 'Nel bin giusto', 'Bin sbagliato', 'Sul tavolo', 'Persi', 'Riconosciuto', 'Scambiato',
                  'Sconosciuto', 'Non visto', 'Senza presa', 'Scivolati', 'Prese a vuoto', 'Dita chiuse in trasporto'], rows)
    confused = collections.Counter((p['code'], p['said']) for p in pieces if p['recognition'] == 'wrong')
    if confused:
        out += ['## Scambi di riconoscimento', '']
        out += table(['Pezzo vero', 'Creduto', 'Volte', 'Su quanti'],
                     [[true, said, n, len(by_code[true])] for (true, said), n in confused.most_common(20)])
    failed = [r for r in judged if r['ok'] != r['pieces'] or r['status'] != 'done']
    if failed:
        out += ['## Ultime corse non perfette', '']
        out += table(['Corsa', 'Scenario', 'Rumore', 'Stampo', 'Per rifarla', 'Stato', 'A posto', 'Log'],
                     [[r['run'], r['scenario'], r['noise'], r.get('mold', ''), '{}:{}:{}'.format(r['scenario'], r['seed'], r.get('mold', '')),
                       r['status'], '{}/{}'.format(r['ok'], r['pieces']), 'logs/{}.log'.format(r['run'])] for r in failed[-25:]])
    text = '\n'.join(out)
    (folder / 'summary.md').write_text(text)
    return text


# ---------------------------------------------------------------- running things

def start_process(command, log_path, environment):
    log = open(log_path, 'w')
    process = subprocess.Popen(['bash', '-c', command], stdout=log, stderr=subprocess.STDOUT, env=environment,
                               start_new_session=True, cwd=environment['BATCH_WORKSPACE'])
    STARTED.add(process)
    return process


def stop_process(process, grace=6.0):
    """The whole process group: a launch leaves its nodes behind otherwise, and they take the next run down."""
    if process is None or process.poll() is not None and not process_group_alive(process):
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        end = time.monotonic() + grace
        while time.monotonic() < end and process_group_alive(process):
            time.sleep(0.2)
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
    STARTED.discard(process)


def process_group_alive(process):
    try:
        os.killpg(process.pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


class Isaac:
    """Headless Isaac Sim, restarted when it hangs: its renderer does, after a while."""

    def __init__(self, args, codes, scene):
        self.args = args
        self.codes = codes
        self.scene = scene                  # for batch_sim.py: where the bins end and the work area is
        self.shared = Path(args.workspace) / 'shared'
        name = '.batch_{}'.format(args.domain)
        self.state = self.shared / name
        self.inside = (Path(args.container_shared) / name) if args.container else self.state
        self.message = 0
        self.running = None
        self.conveyor_wait = 1000000    # the belt must not restart under the robot here: --belt-window measures the round

    def shell(self, command):
        """A command where Isaac Sim lives."""
        if self.args.container:
            return subprocess.run(['docker', 'exec', self.args.container, 'bash', '-c', command], capture_output=True, text=True)
        return subprocess.run(['bash', '-c', command], capture_output=True, text=True)

    def start(self, belt):
        self.stop()
        if self.state.exists():
            shutil.rmtree(self.state, ignore_errors=True)
        self.state.mkdir(exist_ok=True)
        os.chmod(self.state, 0o777)                 # the user of the container is another one
        for name in ('ready', 'heartbeat', 'cmd.json', 'reply.json'):
            (self.state / name).unlink(missing_ok=True)
        shutil.copy(HERE / 'batch_sim.py', self.state / 'batch_sim.py')
        root = Path(self.args.container_shared) if self.args.container else self.shared
        (self.state / 'config.json').write_text(json.dumps(
            dict(self.scene, stage=str(root / self.args.stage), pool=self.args.pool, belt=belt, codes=sorted(self.codes),
                 grip_torque=self.args.grip_torque)))
        run = 'cd /tmp && ROS_DOMAIN_ID={} CONVEYOR_WAIT={} MOLD_FILE={}/mold.txt {}/python.sh {}/batch_sim.py {} > {}/sim.log 2>&1'.format(
            self.args.domain, self.conveyor_wait, self.inside, self.args.isaac_path, self.inside, self.inside, self.inside)
        if self.args.container:
            subprocess.run(['docker', 'exec', '-d', self.args.container, 'bash', '-c', run], check=True)
        else:
            subprocess.Popen(['bash', '-c', run], start_new_session=True)
        end = time.monotonic() + self.args.sim_start_timeout
        while not (self.state / 'ready').exists():
            if time.monotonic() > end:
                self.stop()
                raise SimDown('Isaac Sim did not start in {:.0f} s: see {}/sim.log'.format(self.args.sim_start_timeout, self.state))
            time.sleep(1.0)
        self.running = belt

    def stop(self):
        pid_file = self.state / 'sim.pid'
        if pid_file.exists():
            self.shell('kill -9 {} 2>/dev/null; true'.format(pid_file.read_text().strip()))
            pid_file.unlink(missing_ok=True)
        self.running = None

    def command(self, timeout, **fields):
        """Ask and wait for the answer, watching that Isaac Sim is still alive."""
        self.message += 1
        answer = self.state / 'reply.json'
        answer.unlink(missing_ok=True)
        temporary = self.state / 'cmd.tmp'
        temporary.write_text(json.dumps(dict(fields, id=self.message)))
        os.chmod(temporary, 0o666)
        temporary.replace(self.state / 'cmd.json')
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.check_alive()
            if answer.exists():
                reply = json.loads(answer.read_text())
                if reply.get('id') == self.message:
                    if not reply.get('ok'):
                        raise SimDown(reply.get('error', 'error in Isaac Sim'))
                    return reply
            time.sleep(0.2)
        raise SimDown('no answer to {} in {:.0f} s'.format(fields.get('cmd'), timeout))

    def keep_log(self, path):
        """The log of Isaac Sim goes with its folder at the next start: without it a failure there is a mystery."""
        if (self.state / 'sim.log').exists():
            shutil.copy(self.state / 'sim.log', path)

    def check_alive(self):
        beat = self.state / 'heartbeat'
        if beat.exists() and time.time() - beat.stat().st_mtime > self.args.hang_seconds:
            raise SimDown('Isaac Sim hung: no heartbeat for {:.0f} s'.format(self.args.hang_seconds))


class Batch:

    def __init__(self, args):
        self.args = args
        self.folder = Path(args.out)
        (self.folder / 'logs').mkdir(parents=True, exist_ok=True)
        self.python = HERE.parent
        sorter = sorter_constants(self.python / 'pick_and_place.py')
        self.bins, self.inner, self.reject = sorter['BINS_XY'], sorter['BIN_INNER'], sorter['REJECT_BIN']
        self.models = Path(args.workspace) / 'shared' / 'piece_models.json'
        parts = mold_parts(self.models)
        self.molds = args.molds.split(',') if args.molds else sorted(parts)
        self.mold_bins = {m: mold_bins(self.models, m, len(self.bins), self.reject) for m in self.molds}
        codes = sorted(set(c for m in self.molds for c in self.mold_bins[m]))
        self.isaac = Isaac(args, codes, {'rim_z': sorter['TABLE_TOP_Z'] + sorter['BIN_HEIGHT'], 'work_x': sorter['WORK_X']})
        self.environment = dict(os.environ, ROS_DOMAIN_ID=str(args.domain), BATCH_WORKSPACE=args.workspace,
                                MOLD_FILE=str(self.isaac.state / 'mold.txt'))     # the mold of the run, not the one in production
        self.source = 'source {} && source {}/install/setup.bash && '.format(args.ros_setup, args.workspace)
        self.done = set(r['run'] for r in read_rows(self.folder / 'runs.csv'))
        self.runs_since_start = 0
        self.kept_logs = 0
        for name, fields in (('runs', RUN_FIELDS), ('pieces', PIECE_FIELDS), ('events', EVENT_FIELDS)):
            path = self.folder / '{}.csv'.format(name)
            if not path.exists():
                with open(path, 'w', newline='') as f:
                    csv.DictWriter(f, fields).writeheader()

    def append(self, name, fields, rows):
        with open(self.folder / '{}.csv'.format(name), 'a', newline='') as f:
            writer = csv.DictWriter(f, fields)
            writer.writerows(rows)

    def specs(self):
        """Blocks of runs, round and round the scenarios and the settings: wherever it stops, they are all there."""
        args = self.args
        index = 0
        for item in args.replay:
            index += 1
            scenario, seed, mold = (item.split(':') + [self.molds[0]])[:3]
            yield {'run': 'r{:05d}_{}'.format(index, scenario), 'scenario': scenario, 'noise': args.noise[0],
                   'gripper': args.gripper[0], 'seed': int(seed), 'mold': mold}
        if args.replay:
            return
        rng = random.Random(args.seed)
        while True:
            for scenario in args.scenarios:
                for noise in args.noise:
                    for gripper in args.gripper:
                        for _ in range(1 if scenario == 'scene' and args.block > 1 and not args.smoke else args.block):
                            index += 1
                            seed = rng.randrange(1, 10 ** 8)
                            yield {'run': 'r{:05d}_{}'.format(index, scenario), 'scenario': scenario, 'noise': noise,
                                   'gripper': gripper, 'seed': seed, 'mold': self.molds[index % len(self.molds)]}

    def start_moveit(self, run):
        log_path = self.folder / 'logs' / 'moveit.tmp'
        for attempt in range(self.args.stack_attempts):
            process = start_process(self.source + 'exec ros2 launch xarm_moveit_config xarm5_moveit_isaac.launch.py '
                                    'headless:=true', log_path, self.environment)
            end = time.monotonic() + self.args.stack_timeout
            while time.monotonic() < end:
                text = log_path.read_text(errors='replace')
                if text.count('Configured and activated') >= 3:
                    return process
                if 'Failed loading controller' in text or 'Failed to acquire lock' in text or process.poll() is not None:
                    break
                self.isaac.check_alive()
                time.sleep(0.5)
            stop_process(process)
            time.sleep(2.0)
        raise StackDown('MoveIt did not come up in {} attempts for {}'.format(self.args.stack_attempts, run))

    def one_run(self, spec):
        belt = spec['scenario'] == 'belt'
        if self.isaac.running is None or self.isaac.running != belt or self.runs_since_start >= self.args.restart_every:
            self.isaac.start(belt)
            self.runs_since_start = 0
        self.runs_since_start += 1
        wait = self.args.belt_timeout if belt else 120.0
        (self.isaac.state / 'mold.txt').write_text(spec['mold'] + '\n')     # the belt and the sorter read it
        start = self.isaac.command(wait, cmd='reset', scenario=spec['scenario'], seed=spec['seed'], count=self.args.pieces,
                                   codes=sorted(self.mold_bins[spec['mold']]))['snapshot']
        if not start['pieces']:
            raise SystemExit('nessun pezzo nello scenario {}: scene usa quelli salvati in world.usd'.format(spec['scenario']))
        moveit, extras, sorter = None, [], None
        log_path = self.folder / 'logs' / '{}.log'.format(spec['run'])
        try:
            moveit = self.start_moveit(spec['run'])
            arguments = ''
            watch_file = self.folder / 'logs' / 'joints.tmp'
            watch_file.unlink(missing_ok=True)
            extras.append(start_process(self.source + 'exec python3 {}/watch_joints.py {}'.format(HERE, watch_file),
                                        self.folder / 'logs' / 'watch.tmp', self.environment))
            if spec['noise'] > 0:
                extras.append(start_process(self.source + 'exec python3 {}/depth_noise.py --strength {}'.format(self.python, spec['noise']),
                                            self.folder / 'logs' / 'noise.tmp', self.environment))
                arguments = ' --depth-topic /camera/aligned_depth_to_color/image_noisy'
            if spec['gripper'] == 'driver':
                extras.append(start_process(self.source + 'exec python3 {}/driver_gripper.py'.format(HERE),
                                            self.folder / 'logs' / 'gripper.tmp', self.environment))
            time.sleep(2.0)
            before = self.isaac.command(30.0, cmd='snapshot')['snapshot']
            began = time.monotonic()
            sorter = start_process(self.source + 'exec python3 {}/pick_and_place.py{}'.format(self.python, arguments), log_path, self.environment)
            status = 'done'
            while sorter.poll() is None:
                if time.monotonic() - began > self.args.run_timeout:
                    status = 'timeout'
                    break
                if read_worst(watch_file).get('tracking', 0.0) > MAX_TRACKING:
                    break               # nothing works after that, no use waiting for the script to give up
                self.isaac.check_alive()
                time.sleep(0.5)
            code = sorter.poll()
            stop_process(sorter)
            wall = time.monotonic() - began
            end = self.isaac.command(30.0, cmd='snapshot')['snapshot']
        finally:
            stop_process(sorter)
            for process in extras:
                stop_process(process)
            stop_process(moveit)
        fingers = [abs(v) for finger in end['fingers'] for v in finger]
        worst = read_worst(watch_file)
        if (not end['fingers'] or any(v != v or v > 2.0 for v in fingers) or worst.get('joint4', 0.0) > MAX_JOINT4 or
                worst.get('finger_min', 0.0) < MIN_FINGER or worst.get('tracking', 0.0) > MAX_TRACKING):
            status = 'exploded'
            self.isaac.running = None           # do not trust it for the next run
        return self.record(spec, status, code, start, before, end, wall, log_path)

    def record(self, spec, status, code, start, before, end, wall, log_path):
        log = parse_log(log_path.read_text(errors='replace'))
        rows = judge(start, end, log, self.bins, self.inner, self.mold_bins[spec['mold']])
        tally = collections.Counter(r['outcome'] for r in rows)
        robot = end['t'] - before['t']
        window = ''
        if spec['scenario'] == 'belt':
            window = int(robot <= self.args.belt_window and tally['ok'] == len(rows))
        all_ok = tally['ok'] == len(rows) and bool(rows)
        result = dict(spec, status=status, exit='' if code is None else code, pieces=len(rows), ok=tally['ok'],
                      wrong_bin=tally['wrong_bin'], on_table=tally['on_table'], lost=tally['lost'],
                      robot_s=round(robot, 1), wall_s=round(wall, 1), real_time_factor=round(robot / max(wall, 1e-3), 2),
                      in_window=window, drops=sum(log['drops'].values()),
                      silent_failure=int(code == 0 and not all_ok), false_alarm=int(code not in (0, None) and all_ok),
                      straight_lines=log['counts']['straight_lines'], around=log['counts']['around'],
                      moveit_errors=log['counts']['moveit_errors'])
        self.append('runs', RUN_FIELDS, [result])
        self.append('pieces', PIECE_FIELDS, [dict(r, run=spec['run'], scenario=spec['scenario'], noise=spec['noise'],
                                                  gripper=spec['gripper'], seed=spec['seed'], mold=spec['mold']) for r in rows])
        self.append('events', EVENT_FIELDS, [{'run': spec['run'], 'scenario': spec['scenario'], 'noise': spec['noise'], 'code': c,
                                              'kind': k, 'count': n} for (c, k), n in sorted(log['events'].items())])
        if all_ok and status == 'done' and self.kept_logs >= self.args.keep_logs:
            log_path.unlink(missing_ok=True)    # the good ones only take space, after the first few
        elif all_ok:
            self.kept_logs += 1
        else:
            shutil.copy(self.folder / 'logs' / 'moveit.tmp', self.folder / 'logs' / '{}.moveit.log'.format(spec['run']))
        return result

    def run(self):
        args = self.args
        begun = time.monotonic()
        finished = 0
        print('Risultati in {}\nStampi: {}'.format(self.folder, ', '.join(
            '{} ({})'.format(m, ', '.join(c[-5:] for c in self.mold_bins[m])) for m in self.molds)), flush=True)
        for spec in self.specs():
            if spec['run'] in self.done:
                finished += 1
                continue
            if args.runs and finished >= args.runs or args.hours and time.monotonic() - begun > args.hours * 3600:
                break
            result = None
            for attempt in range(args.run_attempts):
                try:
                    result = self.one_run(spec)
                    break
                except (SimDown, StackDown) as e:
                    print('  {}: {}, attempt {} of {}'.format(spec['run'], str(e).strip().splitlines()[-1][:160], attempt + 1, args.run_attempts), flush=True)
                    self.isaac.keep_log(self.folder / 'logs' / 'isaac_{}_{}.log'.format(spec['run'], attempt + 1))
                    self.isaac.stop()
            if result is None:
                self.append('runs', RUN_FIELDS, [dict(spec, status='infrastructure', exit='', pieces=0, ok=0, wrong_bin=0, on_table=0, lost=0,
                                                      robot_s='', wall_s='', real_time_factor='', in_window='', drops=0, silent_failure=0,
                                                      false_alarm=0, straight_lines=0, around=0, moveit_errors=0)])
            else:
                print('[{}] {} rumore={} pinza={} seed={}: {}/{} a posto, {}, {} s robot, {} s reali'.format(
                    spec['run'][:6], spec['scenario'], spec['noise'], spec['gripper'], spec['seed'], result['ok'], result['pieces'],
                    result['status'], result['robot_s'], result['wall_s']), flush=True)
            finished += 1
            write_report(self.folder, ' '.join(sys.argv))

    def close(self):
        for process in list(STARTED):
            stop_process(process, grace=1.0)
        self.isaac.keep_log(self.folder / 'logs' / 'isaac_last.log')
        self.isaac.stop()
        shutil.rmtree(self.isaac.state, ignore_errors=True)
        for leftover in (self.folder / 'logs').glob('*.tmp'):
            leftover.unlink(missing_ok=True)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hours', type=float, help='stop after this long')
    parser.add_argument('--runs', type=int, help='stop after this many runs')
    parser.add_argument('--smoke', action='store_true', help='one run of each scenario, to see that it all works')
    parser.add_argument('--report', help='write the report of this folder again and exit')
    parser.add_argument('--out', help='results folder, an existing one goes on from where it stopped')
    parser.add_argument('--replay', nargs='+', default=[], metavar='SCENARIO:SEED:MOLD', help='only these runs, e.g. the failed ones of a report')
    parser.add_argument('--molds', help='comma separated, e.g. 940016509,940017433 (default: all of them, one run each in turn)')
    parser.add_argument('--scenarios', default='table,tipped,touching,belt', help='any of ' + ','.join(SCENARIOS))
    parser.add_argument('--noise', default='0,1', help='strengths of depth_noise.py, 0 is the perfect camera')
    parser.add_argument('--gripper', default='sim', help='sim, driver (the action of the real arm over the simulated gripper), or both')
    parser.add_argument('--pieces', type=int, default=5, help='on the table in each run')
    parser.add_argument('--grip-torque', type=float, help='N*m of the gripper motor in Isaac Sim, in place of the 2.2 of world.usd '
                                                          '(N on the piece x 0.11: one motor, two fingers, 0.055 m of lever)')
    parser.add_argument('--pool', type=int, default=3, help='pieces of each code loaded in Isaac Sim')
    parser.add_argument('--block', type=int, default=5, help='runs in a row of each combination')
    parser.add_argument('--seed', type=int, default=1, help='of the sequence of layouts: the same one gives the same runs')
    parser.add_argument('--belt-window', type=float, default=60.0, help='s of robot time a round of the belt should take')
    parser.add_argument('--workspace', default=str(HERE.parents[3]), help='ros_ws')
    parser.add_argument('--stage', default='world.usd', help='in shared/')
    parser.add_argument('--ros-setup', default='/opt/ros/jazzy/setup.bash')
    parser.add_argument('--container', default='isaac-sim', help="of Isaac Sim, '' if it is installed on this machine")
    parser.add_argument('--container-shared', default='/workspace', help='where the container has shared/')
    parser.add_argument('--isaac-path', default='/isaac-sim', help='of Isaac Sim, where it runs')
    parser.add_argument('--domain', type=int, default=78, help='ROS_DOMAIN_ID, not the one you work on')
    parser.add_argument('--run-timeout', type=float, default=900.0, help='s of wall clock for a run')
    parser.add_argument('--restart-every', type=int, default=20, help='runs: its renderer hung after about 45 minutes')
    parser.add_argument('--hang-seconds', type=float, default=180.0)
    parser.add_argument('--sim-start-timeout', type=float, default=600.0)
    parser.add_argument('--belt-timeout', type=float, default=300.0, help='s of wall clock for the belt to bring the pieces')
    parser.add_argument('--stack-timeout', type=float, default=90.0)
    parser.add_argument('--stack-attempts', type=int, default=4)
    parser.add_argument('--run-attempts', type=int, default=3)
    parser.add_argument('--keep-logs', type=int, default=10, help='of runs that went well')
    args = parser.parse_args()
    args.scenarios = args.scenarios.split(',')
    unknown = set(args.scenarios) - set(SCENARIOS)
    if unknown:
        parser.error('unknown scenarios: {}'.format(sorted(unknown)))
    args.noise = [float(n) for n in args.noise.split(',')]
    args.gripper = ['sim', 'driver'] if args.gripper == 'both' else [args.gripper]
    if args.smoke:
        args.noise, args.block, args.runs = [args.noise[-1]], 1, len(args.scenarios)
    if args.replay:
        args.runs = len(args.replay)
    if not args.out:
        args.out = str(Path(args.workspace) / 'batch_results' / time.strftime('%Y%m%d_%H%M'))
    return args


def check_setup(args):
    """Better now than after an hour, on a machine nobody is looking at."""
    workspace = Path(args.workspace)
    problems = []
    for path, what in ((workspace / 'install' / 'setup.bash', 'the workspace is not built (--workspace)'),
                       (workspace / 'shared' / args.stage, 'the stage is not there (--stage)'),
                       (workspace / 'shared' / 'piece_models.json', 'the library of the pieces is not there'),
                       (workspace / 'shared' / 'conveyor_cycle.py', 'the script of the conveyor is not there'),
                       (Path(args.ros_setup), 'ROS 2 is not there (--ros-setup)')):
        if not path.exists():
            problems.append('{}: {}'.format(path, what))
    if args.container:
        found = subprocess.run(['docker', 'exec', args.container, 'test', '-x', args.isaac_path + '/python.sh'], capture_output=True)
        if found.returncode != 0:
            problems.append('container {} not running, or no Isaac Sim in {} (--container, --isaac-path)'.format(args.container, args.isaac_path))
        else:
            found = subprocess.run(['docker', 'exec', args.container, 'test', '-f', '{}/{}'.format(args.container_shared, args.stage)], capture_output=True)
            if found.returncode != 0:
                problems.append('{} in the container is not shared/ (--container-shared)'.format(args.container_shared))
    elif not Path(args.isaac_path, 'python.sh').exists():
        problems.append('no Isaac Sim in {} (--isaac-path)'.format(args.isaac_path))
    if problems:
        sys.exit('Non parto:\n  ' + '\n  '.join(problems))


def stop_when_told():
    """Ctrl-C, kill, or the terminal going away: all of them have to reach the code that stops Isaac Sim and MoveIt. A job
    started with & ignores SIGINT, so it is set here; SIGHUP only if nohup did not ask to ignore it."""
    def stop(number, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:
        signal.signal(signal.SIGHUP, stop)


def main():
    args = parse_arguments()
    if args.report:
        print(write_report(args.report))
        return
    if not (args.hours or args.runs):
        sys.exit('di quanto? --hours, --runs oppure --smoke')
    check_setup(args)
    stop_when_told()
    batch = Batch(args)
    try:
        batch.run()
    except KeyboardInterrupt:
        print('\nFermato: quello che è finito è salvato.', flush=True)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)     # a second Ctrl-C must not stop the stopping
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        batch.close()
        print(write_report(args.out, ' '.join(sys.argv)))


if __name__ == '__main__':
    main()
