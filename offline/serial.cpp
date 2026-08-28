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

    double start = omp_get_wtime();

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

    cout << fixed << setprecision(6);

    cout << "\nSerial Matrix Multiplication Completed\n";
    cout << "Matrix Size : " << N << " x " << N << endl;
    cout << "Time        : " << end - start << " seconds" << endl;

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
