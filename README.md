# Dynamic pathfinding agent with informed search algorithms

---

## Dynamic Pathfinding Agent

A grid-based pathfinding visualizer built in Python using Tkinter (standard library only — no external dependencies required).

### Features

- **Two search algorithms:** A\* and Greedy Best-First Search (GBFS), selectable at runtime
- **Two heuristics:** Manhattan Distance and Euclidean Distance, toggle-able via the GUI
- **8-directional movement** with corner-cutting prevention (cardinal cost = 1.0, diagonal cost = √2)
- **Interactive grid editor:** click or drag to place/remove walls; reposition the Start or Goal node with a single click
- **Random maze generation** with a configurable obstacle density slider (5–70%)
- **Dynamic obstacle mode:** new walls spawn probabilistically while the agent is in motion, triggering automatic real-time replanning only when the current path is blocked
- **Animated visualization:** frontier nodes in yellow, visited nodes in blue, final path in green, agent in orange
- **Live metrics dashboard:** nodes visited, path cost, and execution time (ms) updated on every run and replan

### Requirements

- Python 3.8 or higher
- No third-party packages — Tkinter ships with standard CPython

### How to Run

```bash
python pathfinding_agent.py
```

### Usage

1. Set the grid dimensions using the **Rows / Cols** inputs and click **Apply Resize**
2. Optionally click **Generate** to fill the grid with a random maze at your chosen density, or paint walls manually by clicking and dragging on the grid
3. Switch **Edit Mode** to reposition the Start (cyan) or Goal (purple) node
4. Choose an **Algorithm** (A\* or GBFS) and a **Heuristic** (Manhattan or Euclidean)
5. Toggle **Dynamic Mode** on if you want obstacles to appear while the agent moves
6. Click **▶ Run Pathfinder** — the search expansion animates first, then the agent travels the found path
7. Click **■ Stop** at any time to halt the run

### File Structure

```
pathfinding_agent.py   # entire application — single self-contained file
```

### Algorithm Notes

| Algorithm | Evaluation Function | Optimal? | Notes |
|---|---|---|---|
| A\* | f(n) = g(n) + h(n) | Yes (with admissible heuristic) | Guarantees shortest path |
| GBFS | f(n) = h(n) | No | Faster in open grids, may return suboptimal paths |
