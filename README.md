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

### 1. Serial Matrix Multiplication (`matrix_serial.cpp`)
```bash
g++ -fopenmp matrix_serial.cpp -o matrix_serial.exe
./matrix_serial.exe
```
Enter matrix size $N$ when prompted.

### 2. Parallel Matrix Multiplication with OpenMP (`matrix_parallel.cpp`)
```bash
g++ -fopenmp matrix_parallel.cpp -o matrix_parallel.exe
./matrix_parallel.exe
```
Enter matrix size $N$ and number of threads when prompted.

### 3. Combined Benchmark (`matrix.cpp`)
```bash
g++ -fopenmp -O2 matrix.cpp -o matrix.exe
./matrix.exe
```

### 4. Interactive Web Dashboard
Double-click [`index.html`](index.html) to open in your browser:
- **Serial Mode**: Select **Serial** in the header to run single-threaded matrix multiplication with exact step-by-step logs and result matrix.
- **Parallel Mode**: Select **Parallel** to configure OpenMP threads, observe multi-threaded thread assignment, speedup, efficiency, and real-time computation visualizer.

---

## 📊 Metrics & Formulas

- **Workload per Thread**: $\text{Workload} = \frac{N^3}{p}$
- **Speedup**: $S = \frac{T_{\text{serial}}}{T_{\text{parallel}}}$
- **Efficiency**: $E = \frac{S}{p} \times 100\%$

