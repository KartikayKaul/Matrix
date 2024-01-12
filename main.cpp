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
    A.display("A:-");


    //D.saveMatrix("matrixD");
    matrix<double> slicedMat = A(range(1,4),range(1,4));

    slicedMat.display("slicedMat:-");
    //cout<<range(1,5).size();

    double detA = A.det();
    cout<<endl<<"Determinant of A = "<<detA;


    int *intArray;
    N=3;
    intArray = new int[N*N];
    init2dArray(intArray, N, N);
    matrix<int> B(intArray, N);
    deAlloc(intArray);
    B.display("B:-");

    cout<<"\nDeterminant of B = "<<B.determinant()<<endl;
    return 0;
}