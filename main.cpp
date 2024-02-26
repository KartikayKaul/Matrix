#include<iostream>
#include "matrix.h"
#include<chrono>

using namespace std;
using namespace linear;
using namespace std::chrono;

int main(int arg, char *argv[]) {
    double *array;
    int N = std::atoi(argv[1]);

    // cout<<"N:-"<<N<<endl;
    // array = new double[N*N];
    // init2dRandArray(array, N, N);
    // matrix<double> A;
    // A.updateWithArray(array,N,N);
    // init2dRandArray(array, N, N);
    // matrix<double> B(array, N);
    
    cout<<"\nGenerating two "<<N<<'x'<<N<<" random matrices with double values... ";
    matrix<double> A = randomUniform(N,N);
    matrix<double> B = randomUniform(N,N);
    cout<<"\nAllocation Complete.";

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

    cout<<"\n\n====Benchmark Results====\n";
    cout<<"(Normal Matrix Mul) || Time taken: "<<duration.count() <<" milliseconds\n";       
    cout<<"(SIMD Matrix Mul) || Time taken: "<<duration1.count() <<" milliseconds\n";
    cout<<"(Strassen Matrix Mul) || Time taken: "<<duration2.count() <<" milliseconds\n";
    deAlloc(array);

    cout<<"matmul result == simd result? ";
    (C==D).all(true)? cout<<"true\n":cout<<"false\n";

    cout<<"matmul result == strassen result? ";
    (C==E).all(true) ? cout<<"true\n":cout<<"false\n";
    
    cout<<"simd result == strassen result? ";
    (D==E).all(true) ? cout<<"true\n":cout<<"false\n";


    matrix<double> oowee(3,3,0.);
    oowee.display("OOWEE:-");
    oowee.fillUpperTriangle(4);
    oowee.display("OOWEE OOWEE:-");
    (is_triangular(oowee)) ? cout<<"Owee oowee! It is triangular.\n":cout<<"Noo.\n";
    oowee.fillLowerTriangle(69);
    oowee.display("OOWEE OOWEE CUPCAKKE!:-");

    matrix<int> ana(3,3,2);
    matrix<int> lana(3,3,2);
    (!(ana==lana)).display("ana!=lana:-");
    
    ana(0,1) = 10;
    ana(1,2) = 5;
    ana.display("ana:-");
    (~ana).display("~ana:-");
    // oowee.display("OOWEE OOWEE:-");
    // // type conversion handling
    // std::complex<double> value(5.2,1.5);
    // matrix<std::complex<double>> X(2,3,value);
    // matrix<double> Y;
    // Y = X.reshape(1,6);
    // X.display("X:-");
    // Y.display("Y:-");

    // std::complex<double> value1(1.5, 0.5);
    // matrix<std::complex<double>> G = {{value1},{value1}};
    // G.display();

    // matrix<double> B1 = {{1, 2, 3}};
    // (B1.T()&B1).display("B1' * B1 = ");

    // matrix<double> B2(4);
    // B2.iota(-5);

    // B2.display("B2:-");

    // matrix<double> B3(1,15,2);
    // B3.display("aha");
    // B3.iota();
    // matrix<double> B4;
    // B4 = B3.reshape(3,5);
    
    // B3.display("B3:-");
    // B4.display("B4:-");

    // B4 *= std::complex<double>(2,1);

    // B4.display("B4/= 4.5 :-");

    // matrix<double> C1 = {{1},{2},{0},{3},{4}};
    // C1.display("C1:-");
    // C1.getDims().display("dims(C1):-");
    
    // double dotproduct = (C1.T()&C1).item();
    // cout<<"\n\ninner product of C1 with itself:- "<<dotproduct<<endl;

    // matrix<double> I3 = eye<double>(3);
    // I3.display("I_3x3:-");

    return 0;
}