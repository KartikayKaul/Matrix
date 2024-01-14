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

    matrix<int> C(B);
    C.display("C:-");

    cout<<endl<<B.isComparable(C)<<endl;
    matrix<int> D = B + C;
    D.display("D = B + C:-");
    cout<<"\nDeterminant of B = "<<B.determinant(true)<<endl;

    vector<vector<double>> arr = {{1, 0, 3},
                              {1, 1, 0.5},
                              {2, 0.25, 1}};

    matrix<double> E(arr);
    E.display("E:-");
    cout<<"\ndet(E) = "<<E.determinant();

    matrix<double> newE = E.reshape(9,1);
    newE.display("reshaped E:-");
    matrix<int> dimsnewE = newE.getDims();
    matrix<int> dimsE = E.getDims();

    dimsE.display("dims(E):-");
    dimsnewE.display("dims(newE):-");
    return 0;
}