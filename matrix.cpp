#include <iostream>
#include <omp.h>
#include <iomanip>
using namespace std;

int main()
{
    int N;

    cout << "Name    : Chatakonda Venkata Yaswanth" << endl;
    cout << "Roll No : 2024BCS0245" << endl;

    cout << "Enter matrix size N: ";
    cin >> N;

    // Dynamically allocate matrices
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

    // Initialize matrices
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
    // SERIAL MATRIX MULTIPLICATION
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
    // NUMBER OF THREADS
    // ------------------------------------------------

    int num_threads;

    cout << "Enter number of threads: ";
    cin >> num_threads;

    omp_set_num_threads(num_threads);


    // ------------------------------------------------
    // PARALLEL MATRIX MULTIPLICATION
    // ------------------------------------------------

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
    // CALCULATIONS
    // ------------------------------------------------

    // Total number of multiplication/addition operations
    long long total_work = (long long)N * N * N;

    // Approximate workload handled by each thread
    double workload_per_thread =
        (double)total_work / num_threads;

    // Speedup
    double speedup = serial_time / parallel_time;

    // Efficiency
    double efficiency =
        (speedup / num_threads) * 100.0;


    // ------------------------------------------------
    // DISPLAY RESULTS
    // ------------------------------------------------

    cout << fixed << setprecision(6);

    cout << "\n============================================\n";
    cout << "       PARALLEL MATRIX MULTIPLICATION\n";
    cout << "============================================\n";

    cout << "Matrix Size (N)       : " << N << " x " << N << endl;
    cout << "Number of Threads     : " << num_threads << endl;

    cout << "\nSerial Execution Time : "
         << serial_time << " seconds" << endl;

    cout << "Parallel Execution Time: "
         << parallel_time << " seconds" << endl;

    cout << "Total Work            : "
         << total_work << " operations" << endl;

    cout << "Workload / Thread     : "
         << workload_per_thread << " operations" << endl;

    cout << "Speedup               : "
         << speedup << endl;

    cout << "Efficiency            : "
         << efficiency << " %" << endl;

    cout << "============================================\n";


    // ------------------------------------------------
    // DISPLAY RESULT MATRIX FOR SMALL N
    // ------------------------------------------------

    if (N <= 10)
    {
        cout << "\nResult Matrix:\n";

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                cout << C_parallel[i][j] << " ";
            }
            cout << endl;
        }
    }


    // ------------------------------------------------
    // FREE MEMORY
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