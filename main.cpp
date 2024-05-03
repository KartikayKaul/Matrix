#include<iostream>
#include "matrix.h"
#include<chrono>
#include<Eigen/Dense>

using namespace std;
using namespace linear;
using namespace std::chrono;

int main(int arg, char *argv[]) {
    int N = std::atoi(argv[1]);
    
    using TESTTYPE = double;

    cout<<"\nGenerating two "<<N<<'x'<<N<<" random matrices with TESTTYPE values... ";
    auto alloc_time_start = high_resolution_clock::now();
    matrix<TESTTYPE> A = randomUniform(N,N);
    matrix<TESTTYPE> B = randomUniform(N,N);
    auto alloc_time_end = high_resolution_clock::now();
    auto alloc_dur = duration_cast<milliseconds>(alloc_time_end-alloc_time_start);
    cout<<"\nAllocation Complete. Time taken: "<<alloc_dur.count()<<" milliseconds.\n";
    matrix<TESTTYPE> C, D, E, G, H, J(N,N);
    Eigen::MatrixXd sA = Eigen::MatrixXd::Random(N,N);
    Eigen::MatrixXd sB = Eigen::MatrixXd::Random(N,N);
    Eigen::MatrixXd sC;


    cout<<"\n--:BENCHMARKING:--";
    //benchmarking matrix mul
    cout<<"\n\n MATRIX MULTIPLICATION BENCHMARKING";
    auto start = high_resolution_clock::now();
    C = A&B;
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end-start);

    //benchmarking matrix mul SIMD
    cout<<"\n\n SIMD MATRIX MUL BENCHMARKING";
    auto start1 = high_resolution_clock::now();
    D = matmul_simd(A,B);
    auto end1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(end1-start1);

    //benchmarking matrix mul SIMD
    cout<<"\n\n STRASSEN'S MATRIX MUL BENCHMARKING";
    auto start2 = high_resolution_clock::now();
    E = strassen_multiply(A,B,128);
    auto end2 = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(end2-start2);

    //benchmarking matrix mul parallel strassen
    cout<<"\n\n PARALLEL STRASSEN MATRIX MUL BENCHMARKING";
    auto start4 = high_resolution_clock::now();
    G = para_strassen_multiply(A,B);
    auto end4 = high_resolution_clock::now();
    auto duration4 = duration_cast<milliseconds>(end4-start4);

    //benchmarking matrix mul blocked
    cout<<"\n\n BLOCK MATRIX MUL BENCHMARKING";
    auto start5 = high_resolution_clock::now();
    H = matmul_block(A,B);
    auto end5 = high_resolution_clock::now();
    auto duration5 = duration_cast<milliseconds>(end5-start5);

    //benchmarking matrix mul blocked
    cout<<"\n\n Eigen MATRIX MUL BENCHMARKING";
    auto start6 = high_resolution_clock::now();
    sC = sA*sB;
    auto end6 = high_resolution_clock::now();
    auto duration6 = duration_cast<milliseconds>(end6-start6);
    
    //benchmarking GEMM
    cout<<"\n\n GEMM  BENCHMARKING";
    auto start7 = high_resolution_clock::now();
    matrixproduct(J.begin(), A.begin(), B.begin(), N);
    auto end7 = high_resolution_clock::now();
    auto duration7 = duration_cast<milliseconds>(end7-start7);

    cout<<"\n\n====Benchmark Results====\n";
    cout<<"(Normal Matrix Mul) || Time taken: "<<duration.count() <<" milliseconds\n";       
    cout<<"(SIMD Matrix Mul) || Time taken: "<<duration1.count() <<" milliseconds\n";
    cout<<"(Strassen Matrix Mul) || Time taken: "<<duration2.count() <<" milliseconds\n";
    cout<<"(Parallel Strassen Matrix Mul) || Time taken: "<<duration4.count() <<" milliseconds\n";
    cout<<"(Block Matrix Mul) || Time taken: "<<duration5.count() <<" milliseconds\n";
    cout<<"(Eigen Matrix Mul) || Time taken: "<<duration6.count() <<" milliseconds\n";
    cout<<"( GEMM ) || Time taken: "<<duration7.count() <<" milliseconds\n";

    
    
    cout<<"matmul result == simd result? ";
    (C==D).all(true)? cout<<"true\n":cout<<"false\n";

    cout<<"matmul result == strassen result? ";
    (C==E).all(true) ? cout<<"true\n":cout<<"false\n";
    
    cout<<"simd result == strassen result? ";
    (D==E).all(true) ? cout<<"true\n":cout<<"false\n";

    cout<<"GEMM == normal matmul result? ";
    (J==C).all(true) ? cout<<"true\n":cout<<"false\n";

    cout<<"normal matmul == parallel strassen result? ";
    (C==G).all(true) ? cout<<"true\n":cout<<"false\n";
    
    cout<<"normal matmul == block matmul result? ";
    (C==H).all(true) ? cout<<"true\n":cout<<"false\n";

    cout<<endl;
    return 0;
}