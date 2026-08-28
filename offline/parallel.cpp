#include <iostream>
#include <omp.h>
#include <iomanip>
using namespace std;

int main()
{
    int N, num_threads;

    cout << "Name    : Chatakonda Venkata Yaswanth" << endl;
    cout << "Roll No : 2024BCS0245" << endl;

    cout << "Enter matrix size N: ";
    cin >> N;

    cout << "Enter number of threads: ";
    cin >> num_threads;

    int **A = new int*[N];
    int **B = new int*[N];
    int **C = new int*[N];

    for (int i = 0; i < N; i++)
    {
        A[i] = new int[N];
        B[i] = new int[N];
        C[i] = new int[N];
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i][j] = 1;
            B[i][j] = 1;
            C[i][j] = 0;
        }
    }

    omp_set_num_threads(num_threads);

    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    double end = omp_get_wtime();

    double parallel_time = end - start;

    long long total_work = (long long)N * N * N;

    double workload_per_thread =
        (double)total_work / num_threads;

    cout << fixed << setprecision(6);

    cout << "\n============================================\n";
    cout << "       PARALLEL MATRIX MULTIPLICATION\n";
    cout << "============================================\n";

    cout << "Matrix Size (N)        : "
         << N << " x " << N << endl;

    cout << "Number of Threads      : "
         << num_threads << endl;

    cout << "Parallel Execution Time: "
         << parallel_time << " seconds" << endl;

    cout << "Total Work             : "
         << total_work << " operations" << endl;

    cout << "Workload / Thread      : "
         << workload_per_thread << " operations" << endl;

    cout << "============================================\n";

    if (N <= 10)
    {
        cout << "\nResult Matrix:\n";

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                cout << C[i][j] << " ";
            }
            cout << endl;
        }
    }

    for (int i = 0; i < N; i++)
    {
        delete[] A[i];
        delete[] B[i];
        delete[] C[i];
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
