"""
What the sorter knows of its bins and what it is doing, and the web page that shows it and changes it.

Board keeps, for each bin the robot found: what goes in it (a part number of the mold, the rejects, or nothing), its
maximum, and the pieces that went in, saved in a file so that a restart keeps them. The robot asks it where a piece
goes and waits on it when the bins for that piece are full; the page changes the mold, what each bin is for and its
maximum, and says when a full bin has been changed for an empty one. It also shows the pieces the last look saw on the
belt, known or not, and the one the robot is taking on its way to its bin.

The page is served by WebServer with the Python standard library only: http://<this PC>:8080, updated as things happen
(server-sent events).
"""

import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REJECTS = 'scarti'
FREE = ''
PART_MAX = 20                   # pieces in a bin of parts before it has to be changed, until the page says otherwise
REJECT_MAX = 10                 # in the bin of the rejects: then the robot empties it on the belt
SAME_BIN = 0.05                 # m between a bin found now and one of the saved file, to be the same
SAME_PIECE = 0.002              # m between the piece of an action and one of the last look, to be the same


class Board:

    def __init__(self, path, molds, work_area):
        """molds: {mold: part numbers}, work_area: (x0, x1, y0, y1) where the robot takes the pieces from."""
        self.path = Path(path)
        self.molds = molds
        self.work_area = work_area
        self.lock = threading.Condition()
        self.version = 0
        self.mold = None
        self.bins = []                  # dicts: center, length, width, angle, rim
        self.use = []                   # per bin: a part number, REJECTS or FREE
        self.max = []
        self.contents = []              # per bin: part numbers of what went in, 'unknown' for a reject
        self.waiting = None             # bins the robot waits for, to be changed
        self.status = 'avvio'
        self.action = None              # code, from (x, y), to (bin), phase: pick, carry, drop; piece: its shape if seen
        self.pieces = []                # what the last look saw on the belt: dicts center, length, width, angle, code, left
        self.saved = self.load()
        self.on_changed = None          # called with a bin when the page says it was changed (in Isaac: empty it)
        self.on_mold = None             # called with a mold chosen on the page

    # ------------------------------------------------------------ file

    def load(self):
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {'bins': [], 'use': {}}

    def save(self):
        bins = [{'x': b['center'][0], 'y': b['center'][1], 'max': m, 'contents': c}
                for b, m, c in zip(self.bins, self.max, self.contents)]
        use = self.saved.get('use', {})
        if self.mold:
            use[self.mold] = [{'x': b['center'][0], 'y': b['center'][1], 'use': u} for b, u in zip(self.bins, self.use)]
        self.saved = {'bins': bins, 'use': use}
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.saved, indent=1))
        temporary.replace(self.path)

    @staticmethod
    def saved_for(entries, center):
        near = [e for e in entries if math.hypot(e['x'] - center[0], e['y'] - center[1]) < SAME_BIN]
        return near[0] if near else None

    def changed_now(self):
        self.version += 1
        self.lock.notify_all()

    # ------------------------------------------------------------ the robot

    def set_bins(self, bins):
        """The bins found at the start (piece_vision.Bin): what was in them and their maximum come back from the file."""
        with self.lock:
            self.bins = [{'center': b.center, 'length': b.length, 'width': b.width, 'angle': b.angle, 'rim': b.rim} for b in bins]
            self.max, self.contents = [], []
            for b in self.bins:
                old = self.saved_for(self.saved.get('bins', []), b['center'])
                self.max.append(old['max'] if old else None)
                self.contents.append(list(old['contents']) if old else [])
            self.set_use()
            self.save()
            self.changed_now()

    def set_mold(self, mold):
        with self.lock:
            self.mold = mold
            if self.bins:
                self.set_use()
                self.save()
            self.changed_now()

    def set_use(self):
        """What each bin is for, for this mold: as it was last set on the page, or its parts into the first bins and the
        rejects into the last one."""
        codes = self.molds.get(self.mold, [])
        saved = self.saved.get('use', {}).get(self.mold, [])
        self.use = []
        for i, b in enumerate(self.bins):
            old = self.saved_for(saved, b['center'])
            if old is not None and (old['use'] in codes or old['use'] in (REJECTS, FREE)):
                self.use.append(old['use'])
            elif i == len(self.bins) - 1:
                self.use.append(REJECTS)
            else:
                self.use.append(codes[i] if i < len(codes) else FREE)
        for i, u in enumerate(self.use):
            if self.max[i] is None:
                self.max[i] = REJECT_MAX if u == REJECTS else PART_MAX

    def bins_for(self, code):
        """The bins a piece goes into: those of its part number, those of the rejects for one nobody knows."""
        with self.lock:
            return [i for i, u in enumerate(self.use) if u == (code or REJECTS)]

    def full(self, index):
        with self.lock:
            return len(self.contents[index]) >= self.max[index]

    def target(self, code, stop=None):
        """The bin for a piece: the first of its bins with room. When they are all full, wait until one is changed on the
        page. None if no bin is for it."""
        with self.lock:
            while True:
                bins = [i for i, u in enumerate(self.use) if u == (code or REJECTS)]
                if not bins:
                    return None
                room = [i for i in bins if len(self.contents[i]) < self.max[i]]
                if room:
                    if self.waiting is not None:
                        self.waiting = None
                        self.changed_now()
                    return room[0]
                if self.waiting != bins:
                    self.waiting = bins
                    self.status = 'aspetta: {} piena, va cambiata'.format(
                        ', '.join('cassa {}'.format(i) for i in bins))
                    self.changed_now()
                self.lock.wait(timeout=0.5)
                if stop and stop():
                    return None

    def add(self, index, code):
        with self.lock:
            self.contents[index].append(code or 'unknown')
            self.save()
            self.changed_now()

    def emptied(self, index):
        """The robot emptied this bin itself (the rejects, on the belt)."""
        with self.lock:
            self.contents[index] = []
            self.save()
            self.changed_now()

    def set_pieces(self, pieces):
        """What a look saw on the belt: (piece_vision.Piece, part number or None, True if the robot will take it)."""
        with self.lock:
            self.pieces = [{'center': [float(c) for c in p.center], 'length': float(p.length), 'width': float(p.width),
                            'angle': float(p.angle), 'code': code, 'left': not taken} for p, code, taken in pieces]
            self.changed_now()

    def seen_at(self, center):
        return [p for p in self.pieces if math.hypot(p['center'][0] - center[0], p['center'][1] - center[1]) < SAME_PIECE]

    def set_status(self, text, action=None):
        with self.lock:
            self.status = text
            if action and action.get('from'):
                seen = self.seen_at(action['from'])
                if seen:
                    action = dict(action, piece=seen[0])
            self.action = action
            self.changed_now()

    def set_phase(self, phase):
        with self.lock:
            if self.action:
                self.action = dict(self.action, phase=phase)
                if phase == 'carry':        # lifted: it is no longer on the belt, whether it gets to its bin or not
                    seen = self.seen_at(self.action['from'])
                    self.pieces = [p for p in self.pieces if p not in seen]
                self.changed_now()

    # ------------------------------------------------------------ the page

    def page_set(self, what, index=None, value=None):
        """A change from the page. Returns an error text, or None."""
        with self.lock:
            if what == 'mold':
                if value not in self.molds:
                    return 'stampo sconosciuto'
                if self.on_mold:
                    self.on_mold(value)
                self.mold = value
                if self.bins:
                    self.set_use()
                    self.save()
                self.changed_now()
                return None
            if index is None or not 0 <= index < len(self.bins):
                return 'cassa sconosciuta'
            if what == 'max':
                if not isinstance(value, int) or value < 1:
                    return 'il massimo va da 1 in su'
                self.max[index] = value
            elif what == 'use':
                if value not in self.molds.get(self.mold, []) and value not in (REJECTS, FREE):
                    return 'codice non di questo stampo'
                self.use[index] = value
            elif what == 'changed':
                self.contents[index] = []
                if self.on_changed:
                    self.on_changed(self.bins[index])
            else:
                return 'comando sconosciuto'
            self.save()
            self.changed_now()
            return None

    def snapshot(self):
        with self.lock:
            return {'version': self.version, 'mold': self.mold, 'molds': self.molds, 'status': self.status,
                    'action': self.action, 'waiting': self.waiting, 'work_area': self.work_area, 'time': time.time(),
                    'pieces': self.pieces,
                    'bins': [dict(b, index=i, use=self.use[i], max=self.max[i], contents=self.contents[i])
                             for i, b in enumerate(self.bins)]}


class WebServer:
    """The page, its live updates (/events) and its commands (/set), in a thread of the sorter."""

    def __init__(self, board, port):
        page = (Path(__file__).parent / 'sorter_web.html').read_bytes()

        class Handler(BaseHTTPRequestHandler):

            def log_message(self, *args):
                pass

            def send(self, code, body, kind='application/json'):
                self.send_response(code)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == '/':
                    self.send(200, page, 'text/html; charset=utf-8')
                elif self.path == '/state':
                    self.send(200, json.dumps(board.snapshot()).encode())
                elif self.path == '/events':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    last = -1
                    try:
                        while True:
                            with board.lock:
                                board.lock.wait_for(lambda: board.version != last, timeout=15.0)
                            state = board.snapshot()
                            last = state['version']
                            self.wfile.write('data: {}\n\n'.format(json.dumps(state)).encode())
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                else:
                    self.send(404, b'{}')

            def do_POST(self):
                if self.path != '/set':
                    self.send(404, b'{}')
                    return
                try:
                    request = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                    error = board.page_set(request.get('what'), request.get('bin'), request.get('value'))
                except ValueError:
                    error = 'richiesta non valida'
                self.send(400 if error else 200, json.dumps({'error': error}).encode())

        self.server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def default_state_file(shared):
    return os.environ.get('BINS_STATE_FILE', str(Path(shared) / 'bins_state.json'))
