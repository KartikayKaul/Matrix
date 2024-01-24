#include<iostream>
#include "matrix.h"
#include<chrono>

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
    matrix<double> A(array, N);
    init2dRandArray(array, N, N);
    matrix<double> B(array, N);

    auto start = high_resolution_clock::now();

    //benchmarking matrix mul
    matrix<double> C = A&B;
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end-start);

    cout<<"Time taken: "<<duration.count() <<" milliseconds\n";
    Aggregate anna = {0, "karry"};
    cout<<endl<<anna;
    return 0;
}