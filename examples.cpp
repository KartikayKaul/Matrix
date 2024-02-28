#include<iostream>

#include "matrix.h"
#include "matview.h"

using namespace std;
using namespace linear;

int main() {
    /*
    Matrix Addition
        for addition between two matrices the
        dimensions must match ie they must conform.

        Use `isComparable()` method to see if the
        matrices conform
    */
    matrix<double> A = {{4,1,7},
                     {2,0,5}};
    
    matrix<double> B = {{0,6,5},
                     {3,3,8}};

    (A.isComparable(B))?cout<<"A and B conform.":cout<<"A and B do not conform."; 

    matrix<double> X = A + (4./3.)*B;
    A.display("A:-");
    B.display("B:-");
    X.display("X = [A + (4/3)B]:-");

    /*
    Matrix Multiplication
        We use `&` for matrix multiplication
        and `*` for element-wise product calculation
        for multiplication A&B to be possible <=> A.cols() == B.rows() 
        for elementwise product A*B <=> A.isComparable(B) == true
    */

    // matrix multiplication
    matrix<int> C = {{3, 0},
                     {2, 4}};
    matrix<int> D = {{5, -1},
                     {0, -3}};

    /*
        notice in the below lines
        we are defining a double matrix
        and feeding an integer
        matrix into it.
        matrix library performs type-coversion
        for the numerical types.
        But there might be some loss of information
        that occurs for a few data types
        especially if you try to feed a complex matrix
        into a double matrix it will only supply
        real values into the new double matrix.

        Qualified numerical types:-
            *Integral types
            *fractional types
            *std::complex
    */
    matrix<double> Product = C&D;
    matrix<double> ELEM;
    ELEM = C*D;
    C.display("C:-");
    D.display("D:-");
    
    cout<<"Matrix multiplication:-";
    Product.display("C&D:-");

    //element-wise product
    cout<<"\nElement-wise Product:-";
    ELEM.display("C*D:-");

    
    // Some cool manipulations.
    matrix<char> alphabets(1,26);
    alphabets.iota(65); //A = 65 and a = 97
    alphabets.display("alphabets row matrix:-");

    alphabets /= 2;
    alphabets.display("alphabets /= 2:-");

    // Check if two matrices are equal
    matrix<int> market1 = {{1,2,3,5,7,8}};
    matrix<int> market2 = {{1,2,3,5,7,8}};
    
    //returns a bool matrix of size 1x6
    (market1==market2).display("market1==market2:-");

    //you can also use the queries on bool matrices such as all(value) or isany(value)
    //trying to see if all the elements are true in the resultant boolean matrix
    cout<<(((market1==market2).all(true))?"Unit prices in both markets are same.\n":"Unit prices are not same in both markets.\n");


    // You can try to use boolean operations
    matrix<double> someData = randomUniformInt(10, 10, -10,10);
    someData.display("someData:-");
    matrix<bool> moreThan2 = 2<someData;
    moreThan2.display("lessThan2:-");
    matrix<double> moreThan2Data = someData(moreThan2);
    matrix<double> lessThan2Data = someData(someData<=2);
    moreThan2Data.display("moreThan2Data:-");
    lessThan2Data.display("lessThan2Data:-");
    moreThan2Data.getDims().display();
    //In case of trying to attempt != you can simply do !(A==B)


    vector<vector<double>> vecData = someData.toVector();
    

    return 0;
}