"""
Dynamic Pathfinding Agent  — v2 (fixed)
========================================
Requirements: Python 3.8+  (tkinter ships with standard CPython)

Run:  python pathfinding_agent.py

Fixes vs v1:
  - 8-directional movement  (makes Euclidean heuristic meaningful)
  - Metrics reset on every new run
  - Button highlight groups are fully isolated (no cross-contamination)
  - Start/Goal placement always clears any wall underneath
  - Start == Goal guard with clear error message
  - Density slider shows live % via separate label (not slider label param)
  - Path cost uses actual edge distances (1.0 cardinal, sqrt(2) diagonal)
"""

import tkinter as tk
import heapq, math, random, time
from collections import defaultdict


# ─────────────────────────────────────────────
#  CELL TYPES & COLOURS
# ─────────────────────────────────────────────
EMPTY    = 0
WALL     = 1
START    = 2
GOAL     = 3
FRONTIER = 4
VISITED  = 5
PATH     = 6
AGENT    = 7

COLOUR = {
    EMPTY:    "#0f1117",
    WALL:     "#4b5563",
    START:    "#06b6d4",
    GOAL:     "#a855f7",
    FRONTIER: "#fbbf24",
    VISITED:  "#1e3a8a",
    PATH:     "#10b981",
    AGENT:    "#f97316",
}

GRID_LINE = "#1a1f2e"
BG        = "#070b14"
PANEL_BG  = "#0d1117"
ACCENT    = "#06b6d4"
TEXT      = "#e2e8f0"
DIM       = "#6b7280"


# ─────────────────────────────────────────────
#  PRIORITY QUEUE
# ─────────────────────────────────────────────
class PriorityQueue:
    def __init__(self):
        self._heap    = []
        self._counter = 0

    def push(self, priority, item):
        heapq.heappush(self._heap, (priority, self._counter, item))
        self._counter += 1

    def pop(self):
        _, _, item = heapq.heappop(self._heap)
        return item

    def __len__(self):
        return len(self._heap)


# ─────────────────────────────────────────────
#  HEURISTICS
# ─────────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ─────────────────────────────────────────────
#  8-DIRECTIONAL NEIGHBOURS
#  Cardinal cost = 1.0,  Diagonal cost = sqrt(2)
# ─────────────────────────────────────────────
_SQRT2 = math.sqrt(2)
DIRS = [
    (-1,  0, 1.0),   (1,  0, 1.0),
    ( 0, -1, 1.0),   (0,  1, 1.0),
    (-1, -1, _SQRT2),(-1, 1, _SQRT2),
    ( 1, -1, _SQRT2),( 1, 1, _SQRT2),
]

def get_neighbors(r, c, rows, cols, grid):
    """Yield (nr, nc, cost) for passable 8-directional neighbours.
    Diagonal moves are blocked if either adjacent cardinal cell is a wall
    (prevents clipping through wall corners)."""
    for dr, dc, cost in DIRS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue
        if grid[nr][nc] == WALL:
            continue
        if dr != 0 and dc != 0:          # diagonal: check corner-cutting
            if grid[r][nc] == WALL or grid[nr][c] == WALL:
                continue
        yield nr, nc, cost


# ─────────────────────────────────────────────
#  SEARCH  — A* / GBFS
# ─────────────────────────────────────────────
def search(grid, start, goal, algo="astar", heuristic="manhattan"):
    """
    Returns (path, nodes_visited, steps)
      path          – list[(r,c)] start to goal, or None if unreachable
      nodes_visited – int
      steps         – list of snapshot dicts for step-by-step animation
    """
    if start == goal:
        return [start], 0, []

    rows = len(grid)
    cols = len(grid[0])
    h    = manhattan if heuristic == "manhattan" else euclidean

    pq        = PriorityQueue()
    came_from = {}
    g_score   = defaultdict(lambda: float("inf"))

    g_score[start] = 0.0
    pq.push(h(start, goal), start)

    visited_set  = set()
    frontier_set = {start}
    steps        = []

    while pq:
        current = pq.pop()
        if current in visited_set:
            continue
        frontier_set.discard(current)

        if current == goal:
            path = []
            node = goal
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return path, len(visited_set), steps

        visited_set.add(current)

        for nr, nc, cost in get_neighbors(current[0], current[1], rows, cols, grid):
            nb = (nr, nc)
            if nb in visited_set:
                continue
            tentative_g = g_score[current] + cost
            if tentative_g < g_score[nb]:
                came_from[nb] = current
                g_score[nb]   = tentative_g
                f = (tentative_g + h(nb, goal)) if algo == "astar" else h(nb, goal)
                pq.push(f, nb)
                frontier_set.add(nb)

        steps.append({
            "visited":  frozenset(visited_set),
            "frontier": frozenset(frontier_set),
        })

    return None, len(visited_set), steps


# ─────────────────────────────────────────────
#  APPLICATION
# ─────────────────────────────────────────────
class PathfindingApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Pathfinding Agent")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        # ── settings vars ──
        self.rows = 20
        self.cols = 30
        self._rows_var    = tk.IntVar(value=self.rows)
        self._cols_var    = tk.IntVar(value=self.cols)
        self._density_var = tk.IntVar(value=30)
        self.algo         = tk.StringVar(value="astar")
        self.heuristic    = tk.StringVar(value="manhattan")
        self.edit_mode    = tk.StringVar(value="wall")
        self.dynamic_on   = tk.BooleanVar(value=False)
        self.spawn_rate   = tk.IntVar(value=5)

        # ── grid state ──
        self.start   = (0, 0)
        self.goal    = (self.rows - 1, self.cols - 1)
        self.grid    = self._empty_grid()
        self.display = self._empty_grid()

        # ── animation state ──
        self.running     = False
        self.anim_id     = None
        self.anim_steps  = []
        self.anim_idx    = 0
        self._found_path = []
        self.agent_path  = []
        self.agent_idx   = 0
        self._dyn_grid   = None

        # ── mouse state ──
        self._mouse_down = False
        self._last_cell  = None

        self._build_ui()
        self._build_display()

    # ─── helpers ──────────────────────────────
    def _empty_grid(self):
        return [[EMPTY] * self.cols for _ in range(self.rows)]

    def _path_cost(self, path):
        total = 0.0
        for i in range(1, len(path)):
            r0, c0 = path[i - 1]
            r1, c1 = path[i]
            total += math.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
        return total

    def _set_status(self, msg):
        self.status_var.set(f"● {msg.upper()}")

    def _reset_metrics(self):
        self.m_visited.set("—")
        self.m_cost.set("—")
        self.m_time.set("—")

    # ─── UI construction ──────────────────────
    def _build_ui(self):
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(top, text="PATHFINDER", bg=BG, fg=ACCENT,
                 font=("Courier New", 18, "bold")).pack(side="left")
        tk.Label(top, text="  DYNAMIC NAVIGATION AGENT", bg=BG, fg=DIM,
                 font=("Courier New", 9)).pack(side="left", padx=6)
        self.status_var = tk.StringVar(value="● READY")
        tk.Label(top, textvariable=self.status_var, bg=BG, fg=DIM,
                 font=("Courier New", 9)).pack(side="right")

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=4)

        left = tk.Frame(main, bg=BG, width=215)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        center = tk.Frame(main, bg=BG)
        center.pack(side="left", fill="both", expand=True)

        self._build_left(left)
        self._build_center(center)

    def _section(self, parent, title):
        f = tk.Frame(parent, bg=PANEL_BG, highlightthickness=1,
                     highlightbackground="#1f2937")
        f.pack(fill="x", pady=3)
        tk.Label(f, text=title, bg=PANEL_BG, fg=DIM,
                 font=("Courier New", 8, "bold")).pack(anchor="w", padx=8, pady=(6, 2))
        return f

    def _radio_group(self, parent, var, choices, direction="horizontal"):
        """Fully-isolated radio group — each call creates independent state."""
        frame = tk.Frame(parent, bg=PANEL_BG)
        frame.pack(fill="x", padx=8, pady=(2, 4))

        buttons = []   # list of (tk.Button, value) — local to this group only

        def make_cmd(value):
            def cmd():
                var.set(value)
                for btn, bv in buttons:
                    sel = (bv == var.get())
                    btn.config(bg="#0e2233" if sel else PANEL_BG,
                               fg=ACCENT   if sel else DIM)
            return cmd

        pack_side = "left" if direction == "horizontal" else "top"
        pack_fill = "x"

        for txt, val in choices:
            b = tk.Button(frame, text=txt,
                          font=("Courier New", 8, "bold"),
                          bg=PANEL_BG, fg=DIM, bd=0, cursor="hand2",
                          activebackground="#0e2233", activeforeground=ACCENT,
                          pady=4, relief="flat")
            if direction == "horizontal":
                b.pack(side="left", expand=True, fill="x", padx=1)
            else:
                b.pack(fill="x", padx=1, pady=1)
            buttons.append((b, val))

        # wire commands after list is fully populated
        for b, val in buttons:
            b.config(command=make_cmd(val))

        # apply initial highlight
        cur = var.get()
        for b, val in buttons:
            if val == cur:
                b.config(bg="#0e2233", fg=ACCENT)

        return buttons

    def _build_left(self, left):
        # Grid size
        s = self._section(left, "GRID SIZE")
        for lbl, var in [("ROWS", self._rows_var), ("COLS", self._cols_var)]:
            row = tk.Frame(s, bg=PANEL_BG)
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=lbl, bg=PANEL_BG, fg=DIM,
                     font=("Courier New", 8), width=5, anchor="w").pack(side="left")
            tk.Entry(row, textvariable=var, width=6, bg="#070b14", fg=TEXT,
                     insertbackground=TEXT, font=("Courier New", 10), bd=0,
                     highlightthickness=1,
                     highlightbackground="#374151").pack(side="left")
        tk.Button(s, text="APPLY RESIZE", bg="#111827", fg="#9ca3af",
                  font=("Courier New", 8), relief="flat", cursor="hand2",
                  command=self._apply_resize).pack(fill="x", padx=8, pady=(4, 6))

        # Map generation
        s2 = self._section(left, "MAP GENERATION")
        density_row = tk.Frame(s2, bg=PANEL_BG)
        density_row.pack(fill="x", padx=8, pady=(0, 2))
        tk.Label(density_row, text="OBSTACLE DENSITY:", bg=PANEL_BG, fg=DIM,
                 font=("Courier New", 7)).pack(side="left")
        self._density_lbl = tk.Label(density_row, text="30%", bg=PANEL_BG,
                                     fg=ACCENT, font=("Courier New", 8, "bold"))
        self._density_lbl.pack(side="left", padx=4)

        def on_density(*_):
            self._density_lbl.config(text=f"{self._density_var.get()}%")

        tk.Scale(s2, from_=5, to=70, orient="horizontal",
                 variable=self._density_var, bg=PANEL_BG, fg=TEXT,
                 troughcolor="#1f2937", highlightthickness=0,
                 showvalue=False, command=on_density).pack(fill="x", padx=6)

        bf = tk.Frame(s2, bg=PANEL_BG)
        bf.pack(fill="x", padx=8, pady=(4, 6))
        tk.Button(bf, text="GENERATE", bg="#0e2233", fg=ACCENT,
                  font=("Courier New", 8, "bold"), relief="flat", cursor="hand2",
                  command=self._generate_maze).pack(side="left", expand=True,
                                                    fill="x", padx=(0, 2))
        tk.Button(bf, text="CLEAR", bg="#1a0a0a", fg="#ef4444",
                  font=("Courier New", 8, "bold"), relief="flat", cursor="hand2",
                  command=self._clear_grid).pack(side="left", expand=True, fill="x")

        # Algorithm
        s3 = self._section(left, "ALGORITHM")
        self._radio_group(s3, self.algo,
                          [("A*", "astar"), ("GBFS", "gbfs")])
        tk.Label(s3, text="HEURISTIC", bg=PANEL_BG, fg=DIM,
                 font=("Courier New", 7)).pack(anchor="w", padx=8)
        self._radio_group(s3, self.heuristic,
                          [("MANHATTAN", "manhattan"), ("EUCLIDEAN", "euclidean")])

        # Edit mode
        s4 = self._section(left, "EDIT MODE")
        self._radio_group(s4, self.edit_mode,
                          [("WALL", "wall"), ("START", "start"), ("GOAL", "goal")],
                          direction="horizontal")

        # Dynamic mode
        s5 = self._section(left, "DYNAMIC MODE")
        tk.Checkbutton(s5, text="ENABLE DYNAMIC OBSTACLES",
                       variable=self.dynamic_on, bg=PANEL_BG, fg=TEXT,
                       selectcolor="#0e2233", activebackground=PANEL_BG,
                       font=("Courier New", 8), cursor="hand2").pack(
            anchor="w", padx=8, pady=(0, 2))

        spawn_row = tk.Frame(s5, bg=PANEL_BG)
        spawn_row.pack(fill="x", padx=8, pady=(0, 2))
        tk.Label(spawn_row, text="SPAWN RATE:", bg=PANEL_BG, fg=DIM,
                 font=("Courier New", 7)).pack(side="left")
        self._spawn_lbl = tk.Label(spawn_row, text="5 / 1000", bg=PANEL_BG,
                                   fg="#10b981", font=("Courier New", 8, "bold"))
        self._spawn_lbl.pack(side="left", padx=4)

        def on_spawn(*_):
            self._spawn_lbl.config(text=f"{self.spawn_rate.get()} / 1000")

        tk.Scale(s5, from_=1, to=30, orient="horizontal",
                 variable=self.spawn_rate, bg=PANEL_BG, fg=TEXT,
                 troughcolor="#1f2937", highlightthickness=0,
                 showvalue=False, command=on_spawn).pack(fill="x", padx=6,
                                                         pady=(0, 6))

        # Legend
        s6 = self._section(left, "LEGEND")
        for cell_type, label in [
            (START,    "Start Node"),
            (GOAL,     "Goal Node"),
            (AGENT,    "Agent"),
            (PATH,     "Final Path"),
            (FRONTIER, "Frontier (open)"),
            (VISITED,  "Visited (closed)"),
            (WALL,     "Wall"),
        ]:
            row = tk.Frame(s6, bg=PANEL_BG)
            row.pack(fill="x", padx=8, pady=1)
            tk.Canvas(row, width=12, height=12, bg=COLOUR[cell_type],
                      highlightthickness=0).pack(side="left", padx=(0, 6))
            tk.Label(row, text=label, bg=PANEL_BG, fg="#9ca3af",
                     font=("Courier New", 8)).pack(side="left")
        tk.Frame(s6, bg=PANEL_BG, height=6).pack()

    def _build_center(self, center):
        # Metrics bar
        mf = tk.Frame(center, bg=PANEL_BG, highlightthickness=1,
                      highlightbackground="#1f2937")
        mf.pack(fill="x", pady=(0, 6))
        self.m_visited = tk.StringVar(value="—")
        self.m_cost    = tk.StringVar(value="—")
        self.m_time    = tk.StringVar(value="—")

        for lbl, var, colour in [
            ("NODES VISITED",  self.m_visited, ACCENT),
            ("PATH COST",      self.m_cost,    "#10b981"),
            ("EXEC TIME (ms)", self.m_time,    "#fbbf24"),
            ("ALGORITHM",      self.algo,      "#a855f7"),
            ("HEURISTIC",      self.heuristic, "#f472b6"),
        ]:
            col = tk.Frame(mf, bg=PANEL_BG)
            col.pack(side="left", expand=True, fill="x", padx=8, pady=6)
            tk.Label(col, text=lbl, bg=PANEL_BG, fg=DIM,
                     font=("Courier New", 7)).pack()
            tk.Label(col, textvariable=var, bg=PANEL_BG, fg=colour,
                     font=("Courier New", 14, "bold")).pack()

        # Canvas
        cf = tk.Frame(center, bg=PANEL_BG, highlightthickness=1,
                      highlightbackground="#1f2937")
        cf.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(cf, bg=BG, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=6, pady=6)
        self.canvas.bind("<Configure>",       self._on_resize)
        self.canvas.bind("<ButtonPress-1>",   self._on_mouse_press)
        self.canvas.bind("<B1-Motion>",       self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>",
                         lambda e: setattr(self, "_mouse_down", False))

        # Run button
        self.run_btn = tk.Button(
            center, text="▶   RUN PATHFINDER",
            bg="#0e2233", fg=ACCENT,
            font=("Courier New", 13, "bold"),
            relief="flat", cursor="hand2", pady=10,
            command=self._toggle_run)
        self.run_btn.pack(fill="x", pady=6)

        tk.Label(
            center,
            text="LEFT-CLICK / DRAG = PLACE/REMOVE WALLS  |  "
                 "SWITCH EDIT MODE TO REPOSITION START OR GOAL",
            bg=BG, fg="#374151", font=("Courier New", 8)
        ).pack()

    # ─── canvas helpers ────────────────────────
    @property
    def cell_size(self):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return 20
        return max(4, min(cw // self.cols, ch // self.rows))

    def _on_resize(self, _event):
        self._full_redraw()

    def _full_redraw(self):
        self.canvas.delete("all")
        cs = self.cell_size
        for r in range(self.rows):
            for c in range(self.cols):
                x0, y0 = c * cs, r * cs
                fill = COLOUR.get(self.display[r][c], COLOUR[EMPTY])
                self.canvas.create_rectangle(
                    x0, y0, x0 + cs, y0 + cs,
                    fill=fill, outline=GRID_LINE, width=1)

    def _build_display(self, visited=None, frontier=None,
                       path=None, agent=None):
        """Merge base grid + overlays into self.display, then redraw."""
        g = self.grid
        d = [[g[r][c] for c in range(self.cols)] for r in range(self.rows)]

        if visited:
            for (r, c) in visited:
                if d[r][c] == EMPTY:
                    d[r][c] = VISITED

        if frontier:
            for (r, c) in frontier:
                if d[r][c] in (EMPTY, VISITED):
                    d[r][c] = FRONTIER

        if path:
            for (r, c) in path:
                if d[r][c] not in (START, GOAL, WALL):
                    d[r][c] = PATH

        if agent:
            d[agent[0]][agent[1]] = AGENT

        # start/goal always rendered on top
        sr, sc = self.start
        gr, gc = self.goal
        d[sr][sc] = START
        d[gr][gc] = GOAL

        self.display = d
        self._full_redraw()

    # ─── mouse input ───────────────────────────
    def _cell_from_event(self, event):
        cs = self.cell_size
        r  = event.y // cs
        c  = event.x // cs
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return (r, c)
        return None

    def _on_mouse_press(self, event):
        self._mouse_down = True
        cell = self._cell_from_event(event)
        if cell:
            self._apply_edit(cell)
            self._last_cell = cell

    def _on_mouse_drag(self, event):
        if not self._mouse_down:
            return
        cell = self._cell_from_event(event)
        if cell and cell != self._last_cell:
            self._apply_edit(cell)
            self._last_cell = cell

    def _apply_edit(self, cell):
        if self.running:
            return
        r, c = cell
        mode = self.edit_mode.get()

        if mode == "wall":
            if cell == self.start or cell == self.goal:
                return
            self.grid[r][c] = WALL if self.grid[r][c] != WALL else EMPTY
            self._build_display()

        elif mode == "start":
            if cell == self.goal:
                self._set_status("Cannot place Start on Goal cell")
                return
            self.grid[r][c] = EMPTY   # clear any wall under new start
            self.start = cell
            self._build_display()

        elif mode == "goal":
            if cell == self.start:
                self._set_status("Cannot place Goal on Start cell")
                return
            self.grid[r][c] = EMPTY   # clear any wall under new goal
            self.goal = cell
            self._build_display()

    # ─── grid actions ──────────────────────────
    def _apply_resize(self):
        self._stop()
        try:
            self.rows = max(5, min(50, int(self._rows_var.get())))
            self.cols = max(5, min(80, int(self._cols_var.get())))
        except ValueError:
            pass
        self._rows_var.set(self.rows)
        self._cols_var.set(self.cols)
        self.start = (0, 0)
        self.goal  = (self.rows - 1, self.cols - 1)
        self.grid  = self._empty_grid()
        self._reset_metrics()
        self._build_display()
        self._set_status("Grid resized")

    def _generate_maze(self):
        self._stop()
        density = self._density_var.get()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in (self.start, self.goal):
                    self.grid[r][c] = (WALL if random.random() * 100 < density
                                       else EMPTY)
        self._reset_metrics()
        self._build_display()
        self._set_status("Maze generated")

    def _clear_grid(self):
        self._stop()
        self.grid = self._empty_grid()
        self._reset_metrics()
        self._build_display()
        self._set_status("Grid cleared")

    # ─── run / stop ────────────────────────────
    def _toggle_run(self):
        if self.running:
            self._stop()
        else:
            self._run()

    def _stop(self):
        self.running = False
        if self.anim_id:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None
        self.run_btn.config(text="▶   RUN PATHFINDER",
                            bg="#0e2233", fg=ACCENT)
        self._set_status("Stopped")

    def _run(self):
        # Guard: start == goal
        if self.start == self.goal:
            self._set_status("Start and Goal are the same cell!")
            return

        self._reset_metrics()
        self.running = True
        self.run_btn.config(text="■   STOP", bg="#1a0a0a", fg="#ef4444")
        self._set_status("Searching…")

        t0 = time.perf_counter()
        path, nodes_visited, steps = search(
            self.grid, self.start, self.goal,
            self.algo.get(), self.heuristic.get()
        )
        exec_ms = (time.perf_counter() - t0) * 1000

        if path is None:
            self._set_status("No path found!")
            self._stop()
            return

        self.m_visited.set(str(nodes_visited))
        self.m_cost.set(f"{self._path_cost(path):.2f}")
        self.m_time.set(f"{exec_ms:.2f}")

        self.anim_steps  = steps
        self.anim_idx    = 0
        self._found_path = path
        self._animate_search()

    # ─── search animation ──────────────────────
    def _animate_search(self):
        if not self.running:
            return
        if self.anim_idx < len(self.anim_steps):
            step = self.anim_steps[self.anim_idx]
            self._build_display(visited=step["visited"],
                                frontier=step["frontier"])
            # skip frames so animation finishes in ~1-2 s regardless of grid size
            skip = max(1, len(self.anim_steps) // 200)
            self.anim_idx = min(self.anim_idx + skip, len(self.anim_steps))
            self.anim_id = self.root.after(5, self._animate_search)
        else:
            self._build_display(path=self._found_path)
            if self.dynamic_on.get():
                self.agent_path = list(self._found_path)
                self.agent_idx  = 0
                self._dyn_grid  = [row[:] for row in self.grid]
                self._set_status("Agent moving…")
                self.anim_id = self.root.after(130, self._agent_step)
            else:
                self._set_status("Done!")
                self.running = False
                self.run_btn.config(text="▶   RUN PATHFINDER",
                                    bg="#0e2233", fg=ACCENT)

    # ─── dynamic agent ─────────────────────────
    def _agent_step(self):
        if not self.running:
            return

        path = self.agent_path
        idx  = self.agent_idx

        if idx >= len(path):
            self._set_status("Goal reached!")
            self._stop()
            return

        agent_cell = path[idx]
        remaining  = set(map(tuple, path[idx:]))
        rate       = self.spawn_rate.get()
        obstacle_on_path = False

        # Spawn new walls
        for r in range(self.rows):
            for c in range(self.cols):
                cell = (r, c)
                if cell in (self.start, self.goal, agent_cell):
                    continue
                if (self._dyn_grid[r][c] == EMPTY
                        and random.random() * 1000 < rate):
                    self._dyn_grid[r][c] = WALL
                    self.grid[r][c]      = WALL
                    if cell in remaining:
                        obstacle_on_path = True

        # Replan only when the current path is blocked
        if obstacle_on_path:
            self._set_status("Obstacle detected — replanning…")
            t0 = time.perf_counter()
            new_path, nv, _ = search(
                self._dyn_grid, agent_cell, self.goal,
                self.algo.get(), self.heuristic.get()
            )
            exec_ms = (time.perf_counter() - t0) * 1000

            if new_path is None:
                self._build_display(agent=agent_cell)
                self._set_status("Trapped — no path exists!")
                self._stop()
                return

            # Accumulate metrics across replanning events
            prev = int(self.m_visited.get()) if self.m_visited.get() != "—" else 0
            self.m_visited.set(str(prev + nv))
            self.m_cost.set(f"{self._path_cost(new_path):.2f}")
            self.m_time.set(f"{exec_ms:.2f}")

            self.agent_path = new_path
            self.agent_idx  = 0
            path = new_path
            idx  = 0
            self._set_status("Replanned — continuing…")

        agent_pos = path[idx]
        self.agent_idx += 1
        tail = path[self.agent_idx:] if self.agent_idx < len(path) else []
        self._build_display(path=tail, agent=agent_pos)
        self.anim_id = self.root.after(130, self._agent_step)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    root = tk.Tk()
    root.geometry("1140x740")
    root.minsize(820, 580)
    PathfindingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
