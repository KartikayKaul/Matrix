#include<iostream>
#include "matrix.h"
#include<chrono>
// #include<Eigen/Dense>

using namespace std;
using namespace linear;
using namespace std::chrono;

int main(int arg, char *argv[]) {
    int N = std::atoi(argv[1]);
    
    using TESTTYPE = double;

    cout<<"\nGenerating two "<<N<<'x'<<N<<" random matrices with TESTTYPE values... ";
    auto alloc_time_start = high_resolution_clock::now();
    matrix<TESTTYPE> A = randomNormal(N,N);
    matrix<TESTTYPE> B = randomNormal(N,N);
    auto alloc_time_end = high_resolution_clock::now();
    auto alloc_dur = duration_cast<milliseconds>(alloc_time_end-alloc_time_start);
    cout<<"\nAllocation Complete. Time taken: "<<alloc_dur.count()<<" milliseconds.\n";
    matrix<TESTTYPE> C, D, E, J(N,N);

    cout<<"\n--:BENCHMARKING:--";
    //benchmarking matrix mul
    //cout<<"\n\n MATRIX MULTIPLICATION BENCHMARKING";
    auto start = high_resolution_clock::now();
    C = A&B;
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end-start);

    //benchmarking matrix mul SIMD
    //cout<<"\n\n SIMD MATRIX MUL BENCHMARKING";
    auto start1 = high_resolution_clock::now();
    D = matmul_simd(A,B);
    auto end1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(end1-start1);

    //benchmarking GEMM
    //cout<<"\n\n GEMM  BENCHMARKING";
    auto start7 = high_resolution_clock::now();
    matrixproduct(J.begin(), A.begin(), B.begin(), N);
    auto end7 = high_resolution_clock::now();
    auto duration7 = duration_cast<milliseconds>(end7-start7);

    //benchmarking linear::normmatmul
    //cout<<"\n\n normmatmul  BENCHMARKING";
    auto start2 = high_resolution_clock::now();
    E = linear::normmatmul(A,B);
    auto end2 = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(end2-start2);

    cout<<"\n\n====Benchmark Results====\n";
    cout<<"Matrix type: "<<A.type_s()<<endl;
    cout<<"(Normal Matrix Mul) || Time taken: "<<duration.count() <<" milliseconds\n";       
    cout<<"(SIMD Matrix Mul) || Time taken: "<<duration1.count() <<" milliseconds\n";
    cout<<"( GEMM ) || Time taken: "<<duration7.count() <<" milliseconds\n";
    cout<<"(linear::normmatmul ) || Time taken: "<<duration2.count()<<" milliseconds\n";
    cout<<endl;

    cout<<"\n";
    (D==C).all(true)?cout<<"C==D is true.":cout<<"C==D is false.";
    cout<<"\n";
    return 0;
}