#include<iostream>
#include "matrix.h"
#include "matview.h"

using namespace std;
using namespace linear;

int main() {

    int N = 8192;
    matrix<double> A(N, N, 1.5);
    //A.display("A:-");

    matrixView<double> viewA(A, range(N/2), range(N/2));
    //viewA.display("viewA:-");
    return 0;
}