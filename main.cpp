#include<iostream>
#include "matrix.h"

using namespace std;
using namespace linear;

int main() {
    double *array;
    int N=5;
    array = new double[N*N];
    init2dRandArray(array, N, N);
    matrix<double> A(array, N);

    deAlloc(array);
    (A^2).display();


    //D.saveMatrix("matrixD");

    //cout<<range(1,5).size();
    return 0;
}