#include <iostream>
#include <iomanip>
#include <omp.h>

using namespace std;

int main()
{
    int N;
    cout << "Enter matrix size N (1 to 500): ";
    if (!(cin >> N) || N <= 0 || N > 500) {
        cout << "Invalid matrix size. Please enter a value between 1 and 500.\n";
        return 1;
    }

    int num_threads;
    cout << "Enter number of threads: ";
    if (!(cin >> num_threads) || num_threads <= 0) {
        cout << "Invalid thread count.\n";
        return 1;
    }

    // ------------------------------------------------
    // DYNAMIC MEMORY ALLOCATION
    // ------------------------------------------------
    int **A = new int*[N];
    int **B = new int*[N];
    int **C_serial = new int*[N];
    int **C_parallel = new int*[N];

    for (int i = 0; i < N; i++)
    {
        A[i] = new int[N];
        B[i] = new int[N];
        C_serial[i] = new int[N];
        C_parallel[i] = new int[N];
    }

    // ------------------------------------------------
    // INITIALIZE MATRICES
    // ------------------------------------------------
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i][j] = 1;
            B[i][j] = 1;
            C_serial[i][j] = 0;
            C_parallel[i][j] = 0;
        }
    }

    // ------------------------------------------------
    // 1. SERIAL MATRIX MULTIPLICATION (BASELINE)
    // ------------------------------------------------
    double serial_start = omp_get_wtime();

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                C_serial[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    double serial_end = omp_get_wtime();
    double serial_time = serial_end - serial_start;

    // ------------------------------------------------
    // 2. PARALLEL MATRIX MULTIPLICATION (OPENMP)
    // ------------------------------------------------
    omp_set_num_threads(num_threads);

    double parallel_start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                C_parallel[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    double parallel_end = omp_get_wtime();
    double parallel_time = parallel_end - parallel_start;

    // ------------------------------------------------
    // 3. PERFORMANCE METRICS (SPEEDUP & EFFICIENCY)
    // ------------------------------------------------
    long long total_work = (long long)N * N * N;
    double workload_per_thread = (double)total_work / num_threads;

    double speedup = 0.0;
    double efficiency = 0.0;

    if (parallel_time > 0.000000001) {
        speedup = serial_time / parallel_time;
        efficiency = (speedup / num_threads) * 100.0;
    } else {
        speedup = 1.0;
        efficiency = 100.0;
    }

    // ------------------------------------------------
    // 4. DISPLAY RESULTS
    // ------------------------------------------------
    cout << fixed << setprecision(6);
    cout << "\n============================================\n";
    cout << "       PARALLEL MATRIX MULTIPLICATION       \n";
    cout << "============================================\n";
    cout << "Matrix Size (N)        : " << N << " x " << N << "\n";
    cout << "Number of Threads      : " << num_threads << "\n";
    cout << "--------------------------------------------\n";
    cout << "Serial Execution Time  : " << serial_time << " seconds\n";
    cout << "Parallel Execution Time: " << parallel_time << " seconds\n";
    cout << "Total Operations (N^3) : " << total_work << " ops\n";
    cout << "Workload per Thread    : " << setprecision(2) << workload_per_thread << " ops\n";
    cout << "--------------------------------------------\n";
    cout << "Speedup (Ts / Tp)      : " << setprecision(4) << speedup << " x\n";
    cout << "Efficiency             : " << setprecision(2) << efficiency << " %\n";
    cout << "============================================\n";

    // ------------------------------------------------
    // 5. DISPLAY RESULT MATRIX (FOR SMALL N)
    // ------------------------------------------------
    if (N <= 10)
    {
        cout << "\nResult Matrix (Parallel C):\n";
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                cout << C_parallel[i][j] << " ";
            }
            cout << "\n";
        }
    }

    // ------------------------------------------------
    // 6. CLEANUP MEMORY
    // ------------------------------------------------
    for (int i = 0; i < N; i++)
    {
        delete[] A[i];
        delete[] B[i];
        delete[] C_serial[i];
        delete[] C_parallel[i];
    }

    delete[] A;
    delete[] B;
    delete[] C_serial;
    delete[] C_parallel;

    return 0;
}