
#include "../matrix.h"
#include "Fastor/Fastor.h"
#include<iostream>
#include<chrono>

using namespace std;
using namespace Fastor;
using namespace std::chrono;

int main() {
    const int N = 512;
    Tensor<double, N,N> fA; fA.random();
    Tensor<double, N,N> fB; fB.random();
    Tensor<double, N,N> fC;

    linear::matrix<double> mA = linear::randomUniform(N,N);
    linear::matrix<double> mB = linear::randomUniform(N,N);
    linear::matrix<double> mC, mD, mE;


    cout<<"Fastor matmul vs Matrix matmul:-";
    auto start_f = high_resolution_clock::now();
    fC = fA%fB;
    auto end_f = high_resolution_clock::now();
    auto duration_f = duration_cast<milliseconds>(end_f-start_f);

    auto start_m = high_resolution_clock::now();
    mC = mA&mB;
    auto end_m = high_resolution_clock::now();
    auto duration_m = duration_cast<milliseconds>(end_m-start_m);

    auto start_m_2 = high_resolution_clock::now();
    mD = matmul_block(mA,mB);
    auto end_m_2 = high_resolution_clock::now();
    auto duration_m_2 = duration_cast<milliseconds>(end_m_2-start_m_2);

    auto start_m_3 = high_resolution_clock::now();
    mE = linear::matmul_simd(mA,mB);
    auto end_m_3 = high_resolution_clock::now();
    auto duration_m_3 = duration_cast<milliseconds>(end_m_3-start_m_3);

    cout<<"\nResults:-";
    cout<<"\nlinear::matmul time is "<<duration_m.count()<<" ms.";
    cout<<"\nFastor::matmul time is "<<duration_f.count()<<" ms.";
    cout<<"\nlinear::matmul_block time is "<<duration_m_2.count()<<" ms.";
    cout<<"\nlinear::matmul_simd time is "<<duration_m_3.count()<<" ms.\n";

    cout<<"\bType of mE matrx:- "<<mE.type_s()<<endl;

    return 0;
}