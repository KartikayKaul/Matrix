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
    cout<<"\n\n MATRIX MULTIPLICATION BENCHMARKING";
    auto start = high_resolution_clock::now();
    matrix<double> C = A&B;
    //C.display("C:-");
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end-start);

    // cout<<"\n\n====Matrix Multiply Benchmark Results====\n";
    // cout<<"Time taken: "<<duration.count() <<" milliseconds\n";

    //benchmarking matrix mul SIMD
    cout<<"\n\n SIMD MATRIX MUL BENCHMARKING";
    auto start1 = high_resolution_clock::now();
    matrix<double> D = matmul_simd(A,B);
    //C.display("C:-");
    auto end1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(end1-start1);

    //benchmarking matrix mul SIMD
    cout<<"\n\n STRASSEN'S MATRIX MUL BENCHMARKING\n";
    auto start2 = high_resolution_clock::now();
    matrix<double> E = strassen_multiply(A,B);
    //C.display("C:-");
    auto end2 = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(end2-start2);

    cout<<"\n\n====Benchmark Results====\n";
    cout<<"(Matrix Multiplication) || Time taken: "<<duration.count() <<" milliseconds\n";

    cout<<"(SIMD Matrix Mul) || Time taken: "<<duration1.count() <<" milliseconds\n";
    cout<<"(Strassen Matrix Mul) || Time taken: "<<duration2.count() <<" milliseconds\n";
    deAlloc(array);


    // type conversion handling
    std::complex<double> value(5.2,1.5);
    matrix<std::complex<double>> X(2,3,value);
    matrix<double> Y;
    Y = X;
    X.display("X:-");
    Y.display("Y:-");

    std::complex<double> value1(1.5, 0.5);
    matrix<std::complex<double>> G = {{value1},{value1}};
    G.display();

    matrix<double> UTM = {{1, 0, 0},
                           {2, 3, 0},
                           {1, 2, 1}};
    UTM.display("UTM:-");
    
    (UTM.isSymmetric()) ? cout<<"UTM is symmetric.\n" : cout<<"UTM is not symmetric.\n";

    matrix<double> UTM_rand = upper_triangle_matrix(3);
    UTM_rand.display("UTM_rand:-");
    matrix<double> LTM_rand = tril(3);
    LTM_rand.display("LTM_rand:-");
    
    upper_triangle_matrix(3).display("UTM(1):-");
    utm(3).display("UTM(2):-");
    triu(3).display("UTM(3):-");
    return 0;
}