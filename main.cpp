#include<iostream>
#include "matrix.h"
#include<chrono>

using namespace std;
using namespace linear;
using namespace std::chrono;

int main(int arg, char *argv[]) {
    double *array;
    int N = std::atoi(argv[1]);
    
    cout<<"\nGenerating two "<<N<<'x'<<N<<" random matrices with double values... ";
    auto alloc_time_start = high_resolution_clock::now();
    matrix<double> A = randomUniform(N,N);
    matrix<double> B = randomUniform(N,N);
    auto alloc_time_end = high_resolution_clock::now();
    auto alloc_dur = duration_cast<milliseconds>(alloc_time_end-alloc_time_start);
    cout<<"\nAllocation Complete. Time taken: "<<alloc_dur.count()<<" milliseconds.\n";

    cout<<"\n--:BENCHMARKING:--";
    //benchmarking matrix mul
    cout<<"\n\n MATRIX MULTIPLICATION BENCHMARKING";
    auto start = high_resolution_clock::now();
    matrix<double> C = A&B;
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end-start);

    //benchmarking matrix mul SIMD
    cout<<"\n\n SIMD MATRIX MUL BENCHMARKING";
    auto start1 = high_resolution_clock::now();
    matrix<double> D = matmul_simd(A,B);
    auto end1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(end1-start1);

    //benchmarking matrix mul SIMD
    cout<<"\n\n STRASSEN'S MATRIX MUL BENCHMARKING\n";
    auto start2 = high_resolution_clock::now();
    matrix<double> E = strassen_multiply(A,B);
    auto end2 = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(end2-start2);

    //benchmarking matrix mul blas
    cout<<"\n\n BLAS MATRIX MUL BENCHMARKING\n";
    auto start3 = high_resolution_clock::now();
    matrix<double> F = matmul_blas(A,B);
    auto end3 = high_resolution_clock::now();
    auto duration3 = duration_cast<milliseconds>(end3-start3);

    //benchmarking matrix mul parallel strassen
    cout<<"\n\n PARALLEL STRASSEN MATRIX MUL BENCHMARKING\n";
    auto start4 = high_resolution_clock::now();
    matrix<double> G = para_strassen_multiply(A,B);
    auto end4 = high_resolution_clock::now();
    auto duration4 = duration_cast<milliseconds>(end4-start4);

    cout<<"\n\n====Benchmark Results====\n";
    cout<<"(Normal Matrix Mul) || Time taken: "<<duration.count() <<" milliseconds\n";       
    cout<<"(SIMD Matrix Mul) || Time taken: "<<duration1.count() <<" milliseconds\n";
    cout<<"(Strassen Matrix Mul) || Time taken: "<<duration2.count() <<" milliseconds\n";
    cout<<"(BLAS Matrix Mul) || Time taken: "<<duration3.count() <<" milliseconds\n";
    cout<<"(Parallel Strassen Matrix Mul) || Time taken: "<<duration4.count() <<" milliseconds\n";
    deAlloc(array);

    cout<<"matmul result == simd result? ";
    (C==D).all(true)? cout<<"true\n":cout<<"false\n";

    cout<<"matmul result == strassen result? ";
    (C==E).all(true) ? cout<<"true\n":cout<<"false\n";
    
    cout<<"simd result == strassen result? ";
    (D==E).all(true) ? cout<<"true\n":cout<<"false\n";

    cout<<"blas matmul == normal matmul result? ";
    (C==F).all(true) ? cout<<"true\n":cout<<"false\n";

    cout<<"normal matmul == parallel strassen result? ";
    (C==G).all(true) ? cout<<"true\n":cout<<"false\n";
    

    matrix<double> allones = ones<double>(N);
    matrix<double> allzeros = eye<double>(N);
    matrix<double> resP1 = matmul_simd(allones,allzeros);
    matrix<double> resP2 = allones&allzeros;
    ((resP1 == allones).all(true))?cout<<"\nAll values are ones.":cout<<"\nAll values are not ones.";
    ((resP1 == resP2).all(true))?cout<<"\nboth matmuls giving same results.":cout<<"\nboth matmuls not giving same result.";

    cout<<endl;
    return 0;
}