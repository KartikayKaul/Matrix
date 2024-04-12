# Matrix
 Linear Algebra library in C++

The matrix code as of now is just a copy of the matrix data structure I made in [AbstractDS repo](https://github.com/DrakenWan/Abstract-Data-Structures). I am going to now create a standalone linear algebra library in c++. This is just me experimenting for my [scientific computing](https://www.cs.ucr.edu/~craigs/courses/2023-fall-cs-210/index.html) course. After sufficient operations and functions are added to the library, I will start adding documentation for every operation. You can check the [Updates](#Updates) section to see what changes I've made so far. Some of the earliest changes that are not documented here can be referenced in the commits made to [misc::matrix folder](https://github.com/DrakenWan/Abstract-Data-Structures/tree/master/ADT/miscellaneous/matrix) in AbstractDS repository. For now you can look at  below sections that explain some of the operations briefly and also take a look at [main.cpp](./main.cpp) file which has some experimentation with different operations and also helps show how to work with matrix class in a c++ program.


Matrix library is a templated library so we can have matrix elements of different types. However, it is imperative that the types be numerical types. Preferred numerical type is `double` precision floats. Many of the implemented non-member operations also operate solely on `double` precision floats such as `random`, `zeros`, etc.

Be careful when dealing with `std::complex` matrices. Although I have tested the library with complex matrices and it works fine in many cases but when it comes to performing operations where type conversion might occur, due to there being explicit conversion or implicit, some errors may arise. Moreover, I am actively taking into consideration adding safety measures to handle such cases and accomodate operations for complex numerical operations with the matrices.

## Parallelization and Optimization

Memory alignment has been performed during memory allocation by aligning it based on the `DATA` parameter value of the matrix class template. This significantly helps improve certain algorithms/operations, especially the ones that are cache-friendly, such as `linear::matmul_block` (blocked matrix multiplication). Further enabling `-O3` optimization significantly speeds up operations.

Matrix multiplication operation using OpenACC or OpenMP parallelization based on appropriate compiler flag used. If you do not use the flag, the compiler will by default ignore the parallelization directives added in the code and run sequentially.

In case of `matmul_simd` function, for it to work you have to add in the extra `-mavx`/`-mavx2` flag alongwith `-fopenmp`(optional but will reduce speed) flag. There are directives being used along with simd instructions so it is advised to also include OpenMP flags for the compiler.
On side note, `matmul_simd` performs worse than exclusive OpenMP parallelisation used in `&` and `matmul` operations but does job comparable to OpenMP when compared with the OpenACC parallelization.

One more way to speed up operations is to use `-O3` optimization flag but I have not tested the value of the results on large matrices' operations. `matmul_simd` operation does not have any speedup through O3 optimization so it has not been benchmarked with it.

### Commands
Compiling using OpenMP:-
```bash
g++ main.cpp -o main -fopenmp
```

Compiling using OpenACC:-
```bash
g++ main.cpp -o main -fopenacc
```

Compiling using `-O3` flag:-
```bash
g++ -O3 main.cpp -o main -fopenmp
```

It is recommended that you apply the -fopenmp flag by default. However, even if you do not add the flag the code will execute without errors but there will be no speedup.

For sizes less than 100, parallization is disabled. This value is hard-coded as of now and cannot be changed by an environment variable or input to the `main` function or using a macros.

I have benchmarked the matrix multiplication on matrix size of 1000x1000 for both OpenACC and OpenMP and OpenMP performs much better. I recommend using the `-fopenmp` flag to make use of parallelization but maybe this varies based on the system. So you can try benchmarking by running the [main.cpp](./main.cpp) file in your system.
In my case, OpenMP produces result in ~1000 ms and OpenACC produces same result in ~8500 ms. Without parallelization it takes ~9000 ms. 

### Strassen's algorithm

`strassen_multiply` has been implemented for square matrices whose order is 2^n. The algorithm was sped-up by setting the `base_case_cutoff` parameter to 512 by default. This can be adjusted. It is imperative that the dimensions are in powers of 2 for Strassen algorithm. The algorithms were benchmarked on different matrix sizes of order (2^n). Results are provided below that compare the results of algorithm to matrix sizes.
I am thinking of adding [winograd optimization](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#sub-cubic-algorithms) within Strassen algorithm to test it against with or without this optimization as well as compare with other algorithms.


### Benchmarked results 
Note that the standard `Matrix Multiplication` implementation invokes OpenMP parallelization for matrices of sizes larger than 100. These values are not averaged.

| Matrix Size (n) | Matrix Multiplication | SIMD Matrix Mul | Strassen Matrix Mul | Matrix Mul (-O3) | Strassen (-O3) |
|-------------|-----------------------|-----------------|----------------------| ------------------|------------|
|512     | 250 ms    |  255 ms |  256 ms    | 18 ms | 18 ms
|1024    |  1950 ms   |  2114 ms |  1849 ms   | 123 ms | 137 ms
| 2048    |  15768 ms  |  48661 ms |  13489 ms  | 983 ms | 1044 ms
|  4096    | 132221 ms |  555819 ms |  99023 ms | 10798 ms | 7952 ms
| 8192 |  18.9 min |  81.07 min | 12.26 min | 110.7 s | 64.3 s

The efficiency of Strassen Matrix Multiplication here is high due to the fact we are testing this exclusively on matrices of sizes power of 2. Strassen only works with matrices whose dimensions are powers of 2. It can be made to work with square matrices not of power 2 by padding 0's to augment it's size to nearest power of 2. However, I have not implemented that into it.

You can also tinker the `base_case_cutoff` parameter to test what works better for your input.

## Documentation
 You can go to [Matrix wiki](https://github.com/DrakenWan/Matrix/wiki) to read documentation for example usage and reference.

## Updates
- (commit update timestamp: 110420240359). Added memory alignment for POSIX and non-POSIX system to the internal data of the matrix. This will help optimize some operations. [Example](./Examples/) folder has some example `cpp` programs. One of the code files named [fastorvslinear.cpp](./Examples/fastorvslinear.cpp) compares a pre-existing library - [Fastor's](https://github.com/romeric/Fastor) `Fastor::matmul` operation with my own library's `linear::matmul`. [examples.cpp](./Examples/examples.cpp) is the main file that shows how to work with `linear::matrix`. I am also going to expand on the matrix File operations. I am going to try to add functionality to save matrices in `.mat` MATLAB file format as well as in hdf5 format.
- (commit update timestamp: 02042024). There are a lot of changes that have been made and lot of changes made in this commit and I failed to update the Updates section on some of the previous commits. However, the commit messages and description provide brief and concise description of the concrete changes made. You will notice I have added more functions for matrix multiplication one of them borrows cublass general matrix multiplication function. One of them parallelizes the recursion of strassen algorithm. For this commit, I have added the way to know or display the type of the matrix. I have added a template to define a family of structs `TypeName` for different numerical types (including complex types). Then added a #define macro to paste template specialization code for each numerical type by calling the macros and also added a default "unknown" option if I missed any numerical type. There are two member functions that can be used to achieve this. `print_type` and `type_s`. `print_type` simply prints the type on console. `type_s`, however, returns `char* const` value from the `TypeName` struct. `type_s` is a static member function. It is evaluated at compile time. I have also added functions in the library to generate matrices with random values from normal distribution and also added the `ones` generater function that returns a matrix filled with `1`.
- (commit update timestamp: 19022024). I have improved the code further by handling the numerical type conversions in operations where they are needed. Currently I am thinking of implementing Lazy evaluation in my Matrix code operations for fast runtime in long lines of code where result will be printed afterwards and it will evaluate the expression as a whole. This process might take much longer as I have to think of the design of the library and the way I will apply lazy expressions. I am thinking of adding a default template parameter named `Lazy` of type bool with `false` as default value. If activated it will ensure all operations performed evaluate lazy expressions instead of performing eager evaluation.
- (commit update timestamp: 1402240500). Some experimental code added that does not interfere with other working code. I have added two pointers that point to the `first` and `last` positions in the contiguous memory locations whose initial block is pointed to by `val` in the matrix class. Currently `getTotalMemory` and `swapValues` member functions make use of first and last pointers and the destructor and constructor initialize them when memory is allocated in heap and destroyes them when it goes out of scope.
- (commit update timestamp: 1002240943). I have added [`strassen algorithm`](https://en.wikipedia.org/wiki/Strassen_algorithm) for matrix multiplication. It speeds the multiplication for high order matrices by a significant factor. I have benchmarked the three algorithms for matrix multiplication on matrix sizes starting at 512 since the cutoff value for base case of strassen algorithm is at 512. The benchmark results can be found [here](#benchmarked-results).
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