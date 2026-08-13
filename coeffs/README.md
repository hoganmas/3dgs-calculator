# coeffs

Approximate 1D functions as a **sum of shifted Gaussians** (“gaussian splats”).

Pipeline:

1. Parse a LaTeX (or SymPy) expression  
2. Take its degree-\(N\) Taylor polynomial \(P(x)\) about \(x_0\)  
3. Map monomial coefficients \(\vec p\) through \(R = A^{-1}\) to splat weights  
4. Reconstruct:

\[
P(x)\,e^{-x^2/2} \approx \sum_j W_j\, e^{-(x-c_j)^2/2}
\]

\[
P(x) \approx e^{x^2/2} \sum_j W_j\, e^{-(x-c_j)^2/2}
\]

Hermite conversion is optional here — \(A\) already matches even/odd monomial powers, so weights are simply \(w = R p\).

## Setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Web app (Pyodide)

Browser build under [`../web/`](../web/): SymPy runs in-Wasm via Pyodide—no server-side Python required for evaluation.

```bash
# from repo root
python3 -m http.server 8000
# open http://localhost:8000/web/
```

See [web/README.md](../web/README.md) for details.

## CLI

Print the even/odd \(R\) matrices for a given degree and spacing:

```bash
python -m coeffs -N 4 -e 0.5
```

Decompose a LaTeX expression:

```bash
python -m coeffs '\sin(x)' -N 5 -e 0.25
python -m coeffs 'e^{x}' -N 4 -e 0.2
python -m coeffs '\frac{1}{1-x}' -N 4 -e 0.2
```

### Plotting

Interactive window:

```bash
python -m coeffs '\sin(x)' -N 5 -e 0.25 --plot
```

Save without opening a window:

```bash
python -m coeffs '\sin(x)' -N 5 -e 0.25 --save plots/sin_splats.png --no-show
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `-N` | Taylor / polynomial degree |
| `-e` / `--epsilon` | Spacing between gaussian centers |
| `--x0` | Expansion point (default `0`) |
| `--plot` | Show splat visualization |
| `--save PATH` | Write figure to disk |
| `--no-show` | Skip interactive window |
| `--xmin` / `--xmax` | Plot domain |

The plot has two panels:

- **Top:** individual weighted gaussians (blue +, red −) and their sum vs \(P(x)e^{-x^2/2}\)
- **Bottom:** Taylor \(P(x)\) vs envelope-removed reconstruction (and original \(f(x)\) when available)

## Python API

```python
from coeffs.pipeline import latex_to_gaussians
from coeffs.plot import save_or_show

approx = latex_to_gaussians(r"\sin(x)", N=5, epsilon=0.25)

print(approx.taylor)    # SymPy Taylor polynomial
print(approx.coeffs)    # monomial coefficients p[i] of x^i
print(approx.centers)   # list of (center, weight)

err = approx.error(remove_envelope=True)
print(err["max_rel"])

save_or_show(approx, output="plots/sin_splats.png", show=False)
```

Lower-level pieces live in `gaussians.py`:

```python
from coeffs.gaussians import compute_R, gaussian_weights

R = compute_R(N=4, epsilon=0.5)
w_even, w_odd, mats = gaussian_weights(p, N=4, epsilon=0.5)
```

## Tests

```bash
python -m unittest coeffs.test_approximation coeffs.test_pipeline -v
```

## Layout

| File | Role |
|------|------|
| `gaussians.py` | Build \(A\), \(H\), \(R\); evaluate splat sums / envelope removal |
| `pipeline.py` | LaTeX → Taylor → weights |
| `plot.py` | Splat visualization |
| `__main__.py` | CLI entrypoint |
| `test_*.py` | Unit tests |
