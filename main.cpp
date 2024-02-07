#include<iostream>
#include "matrix.h"
#include<chrono>
#include<complex>

using namespace std;
using namespace linear;
using namespace std::chrono;

struct Aggregate {
    private:
        void countpp() {count++;}
        friend std::ostream& operator<<(std::ostream& os, const Aggregate& obj) {
            return os << "[Aggregate] -> {" << obj.count << "," << obj.name << "}\n";
        }
    public:
        int count;
        std::string name;
};

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

    auto start = high_resolution_clock::now();

    //benchmarking matrix mul
    matrix<double> C = A&B;
    C.display("C:-");
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end-start);

    cout<<"Time taken: "<<duration.count() <<" milliseconds\n";
    
    std::vector<std::vector<std::complex<double>>> complexMatrix;
    // Initialize the complex matrix
    complexMatrix.push_back({std::complex<double>(1.0, 2.0), std::complex<double>(3.0, 4.0)});
    complexMatrix.push_back({std::complex<double>(5.0, 6.0), std::complex<double>(7.0, 8.0)});
    for (const auto& row : complexMatrix) {
        for (const auto& element : row) {
            std::cout << element << ' ';
        }
        std::cout << '\n';
    }

    N=2;
    std::complex<double> *arr;
    arr = new std::complex<double>[N*N];
    init2dRandArray(arr, N, N);
    matrix<std::complex<double>> K(arr, N, N);
    K.display("K:-");
    arr = new std::complex<double>[N*N];
    init2dRandArray(arr, N, N);
    matrix<std::complex<double>> L(arr, N, N);
    deAlloc(arr);
    L.display("L:-");

    (K+L).display("K+L=");

    matrix<double> Z = { {1, 2, 3}, {4, 5, 6}};
    matrix<double> Ab = {{-1.5, 0, -3}, {-4.5, 5, -6.5}};
    (Z+Ab).display("Z+Ab:-");

    matrix<double> AB = randomUniform(2);
    AB.display("AB:-");

    matrix<double> AC = randomUniform(2);
    AC.display("AC:-");
    
    N=3;
    //double *ara = new double[N*N];
    // init2dArray(ara, N, N);
    // matrix<double> BB(ara, N);
    // cout<<endl<<is_triangular(BB);
    // BB.display("BB:-");
    // (!BB).display("transpose(BB):-");
    // (BB.inv()).display("inv(BB):-");
    
    // A 4x3 double matrix 
    matrix<double> pepe = {{1,2,3.01}, {3.2,4,5},{6.5,7,8},{9,0,1}};
    pepe.display("pepe:-");
    pepe.getDims().display("pepe dimensions:-");

    //only real part will contribute in below matrix `nepe`
    matrix<double> nepe = pepe/(std::complex<double>(3,2)); 
    /* NOTE:-
    matrix<double> nepe = pepe/(std::complex<int>(3,2))

    above copy expression will generate error, since, `pepe` is a double matrix
    and complex type has integer elements. So the static_casting will not work.
    There is no workaround for this as of now in the library. 
    Ensure internal elements' data types of a non-primitive numerical 
    data types match with those of the matrix's elements' data type.

    And if it is a custom numerical data type (especially if it is templated), that
    all operator overloads are well defined and all cases for different internal
    data types are considered.
    */ 
    nepe.display("nepe:-");
    nepe = pepe(range(1,4),range(2,3));
    nepe.display("nepe:-");
    return 0;
}