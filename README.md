# PADC-Project
### Parallel & Distributed Computing (PADC) — Parallel Matrix Multiplication with OpenMP & Web Dashboard

An interactive Parallel Matrix Multiplication project featuring a **C++ OpenMP High-Performance Backend** and a **Web Dashboard Visualizer**.

---

## 🚀 Features

- **OpenMP Parallel C++ Engine (`matrix.cpp`)**:
  - Dynamically allocated matrices (supports sizes $N \le 500$).
  - Measures serial vs parallel execution times ($T_{\text{serial}}$ vs $T_{\text{parallel}}$).
  - Computes exact **Speedup** ($T_s / T_p$) and **Efficiency** ($\frac{\text{Speedup}}{p} \times 100\%$).
  - Prevents race conditions and avoids division-by-zero on microsecond executions.

- **Web Dashboard (`index.html`, `style.css`, `app.js`)**:
  - **Custom Dimensions**: Configure Matrix A ($M \times K$) and Matrix B ($K \times N$) up to $500 \times 500$.
  - **Manual Matrix Value Entry**: Interactive editable grids with keyboard arrow navigation and quick toolbars (Fill all, Random, Identity, Zeros).
  - **Animated Visualizer**: Dynamic $A \times B = C$ cell-by-cell matrix visualizer with OpenMP thread mapping.
  - **Performance Charts & Metrics**: Real-time Speedup curves, thread utilization bars, and execution logs.

---

## 🛠️ How to Run

### 1. C++ OpenMP Backend
Compile with OpenMP support using `g++`:
```bash
g++ -fopenmp -O2 matrix.cpp -o matrix.exe
./matrix.exe
```
Enter matrix size $N$ ($1 \le N \le 500$) and the number of threads when prompted.

### 2. Web Dashboard
Double-click [`index.html`](index.html) to open it in any web browser. No server or dependencies required.

---

## 📊 Metrics & Formulas

- **Workload per Thread**: $\text{Workload} = \frac{N^3}{p}$
- **Speedup**: $S = \frac{T_{\text{serial}}}{T_{\text{parallel}}}$
- **Efficiency**: $E = \frac{S}{p} \times 100\%$
