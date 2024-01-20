#include<iostream>
#include "matrix.h"
#include<chrono>

using namespace std;
using namespace linear;
using namespace std::chrono;

int main() {
    double *array;
    int N=500;
    array = new double[N*N];
    init2dRandArray(array, N, N);
    matrix<double> A(array, N);
    init2dRandArray(array, N, N);
    matrix<double> B(array, N);

    

    auto start = high_resolution_clock::now();

    //benchmarking matrix mul
    matrix<double> C = A&B;
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end-start);

    cout<<"Time taken: "<<duration.count() <<" milliseconds\n";
    return 0;
}