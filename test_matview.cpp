#include<iostream>
#include "matrix.h"
#include "matview.h"

using namespace std;
using namespace linear;

int main() {

    matrix<double> A = {{1, 2, 3, 4},
                        {5, 6, 7, 8},
                        {9, 8, 7, 6},
                        {5, 4, 3, 2}};
    A.display("A:-");

    matrixView<double> viewA(A, range(2), range(3));
    viewA.display("viewA:-");
    return 0;
}