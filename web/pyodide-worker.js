/* Pyodide worker: load SymPy + local coeffs package, run parse/decompose. */

const PYODIDE_INDEX = "https://cdn.jsdelivr.net/pyodide/v0.27.5/full/";

const COEFF_FILES = [
  "__init__.py",
  "gaussians.py",
  "pipeline.py",
  "browser.py",
];

let pyodide = null;

async function loadLocalModule(filename) {
  const response = await fetch(`/coeffs/${filename}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch coeffs/${filename}: ${response.status}`);
  }
  return response.text();
}

function shortError(error) {
  const raw = error?.message || String(error);
  const lines = raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const match = lines[i].match(
      /^(?:ValueError|TypeError|RuntimeError|KeyError|Error):\s*(.+)$/
    );
    if (match) return match[1];
  }
  return lines[lines.length - 1] || raw;
}

async function boot() {
  importScripts(`${PYODIDE_INDEX}pyodide.js`);

  postMessage({ type: "status", message: "Downloading Pyodide runtime…" });
  pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX });

  postMessage({ type: "status", message: "Installing numpy, sympy, antlr4…" });
  await pyodide.loadPackage(["numpy", "sympy", "micropip"]);
  const micropip = pyodide.pyimport("micropip");
  await micropip.install("antlr4-python3-runtime==4.11.1");

  postMessage({ type: "status", message: "Mounting coeffs package…" });
  const encoder = new TextEncoder();
  pyodide.FS.mkdirTree("/pkg/coeffs");
  for (const file of COEFF_FILES) {
    const source = await loadLocalModule(file);
    pyodide.FS.writeFile(`/pkg/coeffs/${file}`, encoder.encode(source));
  }

  // Always re-read sources from the server so local edits apply after refresh
  await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/pkg")
from coeffs.browser import decompose, parse_check
`);

  postMessage({ type: "ready", message: "Ready." });
}

async function callPython(fnName, payload) {
  const { latex, N = 4, epsilon = 0.5, x0 = 0 } = payload;
  pyodide.globals.set("latex_in", latex);
  pyodide.globals.set("N_in", N);
  pyodide.globals.set("eps_in", epsilon);
  pyodide.globals.set("x0_in", x0);
  const code =
    fnName === "parse_check"
      ? "parse_check(latex_in)"
      : "decompose(latex_in, N=N_in, epsilon=eps_in, x0=x0_in)";
  const json = await pyodide.runPythonAsync(code);
  return JSON.parse(json);
}

self.onmessage = async (event) => {
  const { id, type, payload } = event.data;
  try {
    if (!pyodide) {
      throw new Error("Pyodide is still starting up.");
    }

    if (type === "parse") {
      const result = await callPython("parse_check", payload);
      postMessage({ id, type: "parse_result", result });
      return;
    }

    if (type === "decompose") {
      postMessage({ type: "status", message: "Running SymPy + decomposition…" });
      const result = await callPython("decompose", payload);
      if (!result.ok) {
        postMessage({ id, type: "error", message: result.error || "Decomposition failed." });
        return;
      }
      postMessage({ id, type: "result", result });
      postMessage({ type: "status", message: "Ready." });
    }
  } catch (error) {
    postMessage({
      id,
      type: "error",
      message: shortError(error),
    });
  }
};

boot().catch((error) => {
  postMessage({
    type: "error",
    message: shortError(error),
  });
});
