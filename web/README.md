# Web calculator (Pyodide + SymPy)

Runs the `coeffs` pipeline **in the browser**:

1. Load Pyodide (CPython on WebAssembly)
2. Install `numpy`, `sympy`, and `antlr4-python3-runtime`
3. Mount the local `coeffs/` package into the Pyodide filesystem
4. Call `decompose()` and plot the resulting 1D gaussian splats

## Run locally

Pyodide must load your Python sources over HTTP (not `file://`).

From the **repo root**:

```bash
python3 -m http.server 8000
```

Open [http://localhost:8000/web/](http://localhost:8000/web/).

First load downloads the Pyodide runtime and packages (can take ~20–60s). After that, decompositions run locally with no backend.

## UI

- **Expression** — LaTeX (`\sin(x)`, `e^{x}`) or plain SymPy (`sin(x)`)
- **N** — Taylor degree
- **ε** — gaussian center spacing
- **x₀** — expansion point

Output: parsed expression, Taylor polynomial, center/weight table, and a 1D splat preview canvas.

## Architecture

| File | Role |
|------|------|
| `index.html` | Calculator shell |
| `app.js` | UI, canvas preview, worker messaging |
| `pyodide-worker.js` | Boots Pyodide, installs deps, runs `coeffs.browser.decompose` |
| `../coeffs/browser.py` | JSON API used inside Pyodide |

## 3D view

After decompose, centers are drawn as camera-facing gaussian billboards in Three.js
(additive blending; green = positive weight, coral = negative). Orbit with the mouse.

The cream curve is the enveloped Taylor target \(P(x)e^{-x^2/2}\).
