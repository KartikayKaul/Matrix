# Matrix
 Linear Algebra library in C++

The matrix code as of now is just a copy of the matrix data structure I made in [AbstractDS repo](https://github.com/DrakenWan/Abstract-Data-Structures). I am going to now create a standalone linear algebra library in c++. This is just me experimenting for my [scientific computing](https://www.cs.ucr.edu/~craigs/courses/2023-fall-cs-210/index.html) course. After sufficient operations and functions are added to the library, I will start adding documentation for every operation. You can check the [Updates](#Updates) section to see what changes I've made so far. Some of the earliest changes that are not documented here can be referenced in the commits made to [misc::matrix folder](https://github.com/DrakenWan/Abstract-Data-Structures/tree/master/ADT/miscellaneous/matrix) in AbstractDS repository. For now you can look at  below sections that explain some of the operations briefly and also take a look at [main.cpp](./main.cpp) file which has some experimentation with different operations and also helps show how to work with matrix class in a c++ program.


Matrix library is a templated library so we can have matrix elements of different types. However, it is imperative that the types be numerical types. Preferred numerical type is `double` precision floats. Many of the implemented non-member operations also operate solely on `double` precision floats such as `random`, `zeros`, etc.

Be careful when dealing with `std::complex` matrices. Although I have tested the library with complex matrices and it works fine in many cases but when it comes to performing operations where type conversion might occur, due to there being explicit conversion or implicit, some errors may arise. Moreover, I am actively taking into consideration adding safety measures to handle such cases and accomodate operations for complex numerical operations with the matrices.

## Parallelization and Optimization

Matrix multiplication operation using OpenACC or OpenMP parallelization based on appropriate compiler flag used. If you do not use the flag, the compiler will by default ignore the parallelization directives added in the code and run sequentially.


In case of `matmul_simd` function, for it to work you have to add in the extra `-mavx` flag alongwith `-fopenmp` flag. There are directives being used along with simd instructions so it is advised to also include OpenMP flags for the compiler.
On side note, `matmul_simd` performs worse than exclusive OpenMP parallelisation used in `&` and `matmul` operations but does job comparable to OpenMP when compared with the OpenACC parallelization.

Compiling using OpenMP:-
```bash
g++ main.cpp -o main -fopenmp
```

Compiling using OpenACC:-
```bash
g++ main.cpp -o main -fopenacc
```
For sizes less than 100, parallization is disabled. This value is hard-coded as of now and cannot be changed by an environment variable or input to the `main` function or using a macros.

I have benchmarked the matrix multiplication on matrix size of 1000x1000 for both OpenACC and OpenMP and OpenMP performs much better. I recommend using the `-fopenmp` flag to make use of parallelization but maybe this varies based on the system. So you can try benchmarking by running the [main.cpp](./main.cpp) file in your system.
In my case, OpenMP produces result in ~1000 ms and OpenACC produces same result in ~8500 ms. Without parallelization it takes ~9000 ms. 

### Strassen's algorithm

`strassen_multiply` has been implemented for square matrices whose order is 2^n. The algorithm was sped-up by setting the `base_case_cutoff` parameter to 512 by default. This can be adjusted. Recommended values are in powers of 2. The algorithms were benchmarked on different matrix sizes of order (2^n). Results are provided below that compare the results of algorithm to matrix sizes.


### Benchmarked results 
Note that the standard `Matrix Multiplication` implementation invokes OpenMP parallelization for matrices of sizes larger than 100.

| Matrix Size | Matrix Multiplication | SIMD Matrix Mul | Strassen Matrix Mul |
|-------------|-----------------------|-----------------|----------------------|
| n = 512     | Time taken: 250 ms    | Time taken: 255 ms | Time taken: 256 ms    |
| n = 1024    | Time taken: 1950 ms   | Time taken: 2114 ms | Time taken: 1849 ms   |
| n = 2048    | Time taken: 15768 ms  | Time taken: 48661 ms | Time taken: 13489 ms  |
| n = 4096    | Time taken: 132221 ms | Time taken: 555819 ms | Time taken: 99023 ms |


## Documentation
 You can go to [Matrix wiki](https://github.com/DrakenWan/Matrix/wiki) to read documentation for example usage and reference.

## Updates
- (commit update timestamp: 1002240943). I have added [`strassen algorithm`](https://en.wikipedia.org/wiki/Strassen_algorithm) for matrix multiplication. It speeds the multiplication for high order matrices by a lot of factor. I have benchmarked the three algorithms for matrix multiplication on matrix sizes starting at 512 since the cutoff value for base case of strassen algorithm is at 512. The benchmark results can be found [here](#benchmarked-results).
- (commit update timestamp: 0902240138). You can try working with `complex` matrices and experiment. If any errors arise, please raise them in issues section. I have tried to make sure that `std::complex` matrices are handled properly handled while type conversion arises.
- (commit update timestamp: 0902240117).
   - Added `matmul_simd` that uses AVX instructions. Performed benchmarking (sortof) in [main.cpp](./main.cpp) file.
   - Increased implementation in operator= overload to handle type conversions. NOTE: Currently handling type-conversions with `std:complex` type will seem to generate errors that  I am working on. 
- (commit update timestamp: 9702240400). I have added parallelization using OpenACC
   and OpenMP together. 
- (commit update zeitstamp: 0602241723). I have reduced the logic of operator== overload from 'checking equality of each element' to logic of `isComparable` method. This is much practical logic than the previous one I had implemented.
- (commit update timestampt: 0402241621). I have done a lot of mistakes in non-member matrix operations such as not taking into consideration the constant parameters. This is raising very silly little errors. I will correct them asap. The operations are not working because I am trying to modify constant matrices.
- (commit update timestamp: 1601242153). I am overhauling the entire Matrix operation prototypes as well as implementations. So some of the functions might be missing their definitions. I will add them asap else. The functions that have their definitions present are working correctly.
- (timestamp: 1401240408) `reshape` method has been implemented with correct logic. Working fine. Will keep testing for boundary and special cases.
- Currently working on `determinant()` method. Even though this is the most useless linear algebra function but still I am working on it. I will try to create LU decomposition function and try to calculate the determinant using `det(A) = det(L) * det(U) if A = LU`.