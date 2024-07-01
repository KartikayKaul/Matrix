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
    
    double sum1=0,sum2=0, sum3=0, sum4=0;
    int c1=0, c2=0, c3=0, c4=0;
    const int ITERS=100;
    cout<<"\nProgress: ";
    for(int i=0; i<ITERS; ++i) {
        //benchmarking matrix mul
        auto start = high_resolution_clock::now();
        C = A&B;
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end-start);

        //benchmarking matrix mul SIMD
        auto start1 = high_resolution_clock::now();
        D = matmul_simd(A,B);
        auto end1 = high_resolution_clock::now();
        auto duration1 = duration_cast<milliseconds>(end1-start1);

        //benchmarking GEMM
        auto start7 = high_resolution_clock::now();
        matrixproduct(J.begin(), A.begin(), B.begin(), N);
        auto end7 = high_resolution_clock::now();
        auto duration7 = duration_cast<milliseconds>(end7-start7);

        //benchmarking linear::normmatmul
        auto start2 = high_resolution_clock::now();
        E = linear::normmatmul(A,B);
        auto end2 = high_resolution_clock::now();
        auto duration2 = duration_cast<milliseconds>(end2-start2);

        sum1+= duration.count(); sum2+= duration1.count(); sum3+= duration7.count(); sum4+= duration2.count();
        ++c1; ++c2; ++c3; ++c4;

        if(i%10==0) cout<<"=";
    }

    cout<<"\n\n====Benchmark Results====\n";
    cout<<"Matrix type: "<<A.type_s()<<endl;
    cout<<"(Normal Matrix Mul) || Time taken: "<<sum1/c1 <<" milliseconds\n";       
    cout<<"(SIMD Matrix Mul) || Time taken: "<<sum2/c2 <<" milliseconds\n";
    cout<<"( GEMM ) || Time taken: "<<sum3/c3 <<" milliseconds\n";
    cout<<"(linear::normmatmul ) || Time taken: "<<sum4/c4<<" milliseconds\n";
    cout<<endl;

    cout<<"\n";
    (D==C).all(true)?cout<<"C==D is true.":cout<<"C==D is false.";
    cout<<"\n";
    
    
    matrix<double> M1 = linear::randomUniform(3);
    //cout<<"Matrix M1:-\n"<<M1;
    matrix<double> L, U;
    ludecomp(M1, L, U);
    cout<<is_triangular(L)<<endl;
    cout<<is_triangular(U)<<endl;
    cout<<"M1:-\n"<<M1;
    cout<<"U:-\n"<<U<<"\nL:-\n"<<L;

    return 0;
}