#include<iostream>

#include "matrix.h"

using namespace std;
using namespace linear;

int main() {
    matrix<double> aka(8192);
    cout<<aka.getTotalMemory()<<endl;
    return 0;
}