# Matrix
 Linear Algebra library in C++

The matrix code as of now is just a copy of the matrix data structure I made in [AbstractDS repo](https://github.com/DrakenWan/Abstract-Data-Structures). I am going to now create a standalone linear algebra library in c++. This is just me experimenting for my [scientific computing](https://www.cs.ucr.edu/~craigs/courses/2023-fall-cs-210/index.html) course. After sufficient operations and functions are added to the library, I will start adding documentation for every operation. You can check the [Updates](#Updates) section to see what changes I've made so far. Some of the earliest changes that are not documented here can be referenced in the commits made to [misc::matrix folder](https://github.com/DrakenWan/Abstract-Data-Structures/tree/master/ADT/miscellaneous/matrix) in AbstractDS repository. For now you can look at  below sections that explain some of the operations briefly and also take a look at [main.cpp](./main.cpp) file which has some experimentation with different operations and also helps show how to work with matrix class in a c++ program.


Matrix library is a templated library so we can have matrix elements of different types. However, it is imperative that the types be numerical types. Preferred numerical type is `double` precision floats. Many of the implemented non-member operations also operate solely on `double` precision floats such as `random`, `zeros`, etc.

Be careful when dealing with `std::complex` matrices. Although I have tested the library with complex matrices and it works fine in many cases but when it comes to performing operations where type conversion might occur, due to there being explicit conversion or implicit, some errors may arise. Moreover, I am actively taking into consideration adding safety measures to handle such cases and accomodate operations for complex numerical operations with the matrices.

## Parallelization
We are using OpenMP parallelization for matrix multiplication operation. If in general the matrix dimensions exceed 100 in your code then you should use `-fopenmp` flag for your compiler. Without using this flag the parallization won't work. For example say you ran your code in `main.cpp` file then you can do `g++ main.cpp -o main -fopenmp`. For sizes less than 100, parallization is disabled. This value is hard-coded as of now and cannot be changed by an environment variable or input to the `main` function or using a macros.

## Documentation
 You can go to [Matrix wiki](https://github.com/DrakenWan/Matrix/wiki) to read documentation for example usage and reference.

## Updates
- (commit update zeitstamp: 0602241723). I have reduced the logic of operator== overload from 'checking equality of each element' to logic of `isComparable` method. This is much practical logic than the previous one I had implemented.
- (commit update timestampt: 0402241621). I have done a lot of mistakes in non-member matrix operations such as not taking into consideration the constant parameters. This is raising very silly little errors. I will correct them asap. The operations are not working because I am trying to modify constant matrices.
- (commit update timestamp: 1601242153). I am overhauling the entire Matrix operation prototypes as well as implementations. So some of the functions might be missing their definitions. I will add them asap else. The functions that have their definitions present are working correctly.
- (timestamp: 1401240408) `reshape` method has been implemented with correct logic. Working fine. Will keep testing for boundary and special cases.
- Currently working on `determinant()` method. Even though this is the most useless linear algebra function but still I am working on it. I will try to create LU decomposition function and try to calculate the determinant using `det(A) = det(L) * det(U) if A = LU`.