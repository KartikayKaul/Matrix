# Matrix
 Linear Algebra library in C++

The matrix code as of now is just a copy of the matrix data structure I made in [AbstractDS repo](https://github.com/DrakenWan/Abstract-Data-Structures). I am going to now create a standalone linear algebra library in c++. This is just me experimenting for my [scientific computing](https://www.cs.ucr.edu/~craigs/courses/2023-fall-cs-210/index.html) course. After sufficient operations and functions are added to the library, I will start adding documentation for every operation. You can check the [Updates](#Updates) section to see what changes I've made so far. Some of the earliest changes that are not documented here can be referenced in the commits made to [misc::matrix folder](https://github.com/DrakenWan/Abstract-Data-Structures/ADT/miscellaneous/matrix/) in AbstractDS repository. For now you can look at  below sections that explain some of the operations briefly and also take a look at [main.cpp](./main.cpp) file which has some experimentation with different operations and also helps show how to work with matrix class in a c++ program.

## Operations 

### Functions
Following list has some standalone functions/operations that are present in the library. These functions are not member functions of matrix class.
Since Matrix linear algebra library makes use of templates it is advised to always follow the function name, in the call, with `<datatype>`.
* `eye()`: calling `eye<double>(3)` will yield a 3x3 identity matrix with `double` values
* `is_triangular()`: returns `true` if a matrix is triangular
* `diagonal()`: calling `diagonal<double>(3,0.5)` will yield a 3x3 diagonal matrix with  

## Updates
- (timestamp: 1401240408) `reshape` method has been implemented with correct logic. Working fine. Will keep testing for boundary and special cases.
- Currently working on `determinant()` method. Even though this is the most useless linear algebra function but still I am working on it. I will try to create LU decomposition function and try to calculate the determinant using `det(A) = det(L) * det(U) if A = LU`.