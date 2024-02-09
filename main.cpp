#include<iostream>
#include "matrix.h"
#include<chrono>

using namespace std;
using namespace linear;
using namespace std::chrono;

int main(int arg, char *argv[]) {
    double *array;
    int N = std::atoi(argv[1]);

    cout<<"N:-"<<N<<endl;
    array = new double[N*N];
    init2dRandArray(array, N, N);
    matrix<double> A;
    A.updateWithArray(array,N,N);
    init2dRandArray(array, N, N);
    matrix<double> B(array, N);

    

    //benchmarking matrix mul
    cout<<"\n\n MATRIX MULTIPLICATION BENCHMARKING\n\n";
    auto start = high_resolution_clock::now();
    matrix<double> C = A&B;
    //C.display("C:-");
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end-start);

    // cout<<"\n\n====Matrix Multiply Benchmark Results====\n";
    // cout<<"Time taken: "<<duration.count() <<" milliseconds\n";

    //benchmarking matrix mul SIMD
    cout<<"\n\n SIMD MATRIX MUL BENCHMARKING\n\n";
    auto start1 = high_resolution_clock::now();
    matrix<double> D = matmul_simd(A,B);
    //C.display("C:-");
    auto end1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(end1-start1);

    cout<<"\n\n====Benchmark Results====\n";
    cout<<"(Matrix Multiplication) || Time taken: "<<duration.count() <<" milliseconds\n";

    cout<<"(SIMD Matrix Mul) || Time taken: "<<duration1.count() <<" milliseconds\n";
    deAlloc(array);

    // type conversion handling
    std::complex<double> value(5.2,1.5);
    matrix<std::complex<double>> X(2,3,value);
    matrix<double> Y;
    Y = X;
    X.display("X:-");
    Y.display("Y:-");
    return 0;
}