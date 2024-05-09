# Matrix
A Linear Algebra library in C++. Matrix is a class template library written for fast linear algebra operations. 


The matrix code as of now is just a copy of the matrix data structure I made in [AbstractDS repo](https://github.com/DrakenWan/Abstract-Data-Structures). I am going to now create a standalone linear algebra library in c++. This is just me experimenting for my [scientific computing](https://www.cs.ucr.edu/~craigs/courses/2023-fall-cs-210/index.html) course. After sufficient operations and functions are added to the library, I will start adding documentation for every operation. You can check the [Updates](#Updates) section to see what changes I've made so far. Some of the earliest changes that are not documented here can be referenced in the commits made to [misc::matrix folder](https://github.com/DrakenWan/Abstract-Data-Structures/tree/master/ADT/miscellaneous/matrix) in AbstractDS repository. For now you can look at  below sections that explain some of the operations briefly and also take a look at [main.cpp](./main.cpp) file which has some experimentation with different operations and also helps show how to work with matrix class in a c++ program.

Matrix library makes use of matrix class template so we can have matrix elements of different types. However, it is imperative that the types be numerical types. Preferred numerical type is `double` precision floats. Checking the type of the type parameter is done at compile time using `linear::is_numeric_v` type trait. It includes all possible numerical types including `std::complex`.

Be careful when dealing with `std::complex` matrices. Although I have tested the library with complex matrices and it works fine in many operations but in some operations there may arise some issues. You can report them in [issues section](https://github.com/DrakenWan/Matrix/issues). Moreover, I am actively taking into consideration adding safety measures to handle such cases and accomodate operations for complex numerical operations with the matrices.


## Parallelization and Optimization

I have haphazardly added code for `GEMM` inspired by [`BLIS`](https://www.cs.utexas.edu/users/flame/pubs/blis1_toms_rev3.pdf) framework and code is sourced from [here](https://stackoverflow.com/a/35637007). I want to add more Blocksize specializations for different types my `matrix` class supports and possibly put the whole GEMM kernel in entirely different header file and then combine both of them in a singular header file. I will probably do that. Anyway...

Matrix multiplication operation `linear::matmul` (or `operator&`) makes use of the gemm implementation along with OpenMP parallelization based on appropriate compiler flag used. If you do not use the flag, the compiler will by default ignore the parallelization directives added in the code and run sequentially.

In case of `linear::matmul_simd` function, for it to work you have to add in the extra `-mavx`/`-mavx2` flag alongwith `-fopenmp`(optional but will reduce speed) flag. There are directives being used along with simd instructions so it is advised to also include OpenMP flags for the compiler. `linear::matmul_simd`only works on `int`, `double` and `float` matrices. You can also use `-mfma` if your architecture has [*fused multiply add*](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation#fused-multiply%E2%80%93add). It only works on matrices whose internal dimensions for matrix multiplication are multiples of 8. The appropriate exception handling is done for cases where it is not so.
On side note, `linear::matmul_simd` performs worse than `linear::matmul` and it can be fast for matrices of sizes around 128 and 512 but beyond and below it performs badly.

There is also a standalone GEMM function `linear::matrixproduct` that actually performs faster than `linear::matmul`. However, there are some limitations. It works on square matrices of type `double`, `float` or `int` only. It is demonstrated below:-
```cpp
const int N = 512;
linear::matrix<double> A(N,N);
linear::matrix<double> B(N,N);
linear::matrix<double> C(N,N);

// use .begin() method to return DATA* pointers to the data of matrix.
linear::matrixproduct(C.begin(), A.begin(), B.begin(), N);
```
This usually produces matrix much faster than `linear::matmul` as will be shown in the [Benchmark](#benchmarked-results) section below.

Point to note is that I have tested it only on gcc compiler. Optimizations on other compilers such as icx have not been tested. In near future, I will also test this on MSVC and ICC as well. Currently plan is to make the code as portable as possible and make use of `std` functions to achieve so if possible.

### Commands
Compiling using OpenMP:-
```bash
g++ main.cpp -o main -fopenmp -O3 
```

Add `-fma` or `-mavx` flags for simd intrinsics based on the internal vector architecture you have.
```bash
g++ main.cpp -o main -fopenmp -O3 -mavx -mfma
```

### Strassen's algorithm

`linear::strassen_multiply` has been implemented for square matrices whose order is 2^n. The algorithm was sped-up by setting the `base_case_cutoff` parameter to 512 by default. This can be adjusted. It is imperative that the dimensions are in powers of 2 for Strassen algorithm. The algorithms were benchmarked on different matrix sizes of order (2^n). Results are provided below that compare the results of algorithm to matrix sizes.
I am thinking of adding [winograd optimization](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#sub-cubic-algorithms) within Strassen algorithm to test it against with or without this optimization as well as compare with other algorithms.


### Benchmarked results 
Note that the standard `Matrix Multiplication` implementation invokes OpenMP parallelization for matrices of sizes larger than 100. These time values are averaged over 1000 iterations and are run with `-O3` level optimization in with gcc compiler. The standard followed is from C++17. For `Fastor::matmul` the Tensors could only be defined upto 512. All the matrices are of dimensions in power of 2.

Different `matmul` implementations in the columns of the benchmark table are described in the following list:-
* `linear::matmul` is the default matrix multiplication in my library which uses [`BLIS`]((https://www.cs.utexas.edu/users/flame/pubs/blis1_toms_rev3.pdf)) inspired GEMM.
* `Fastor::matmul` is the matrix multiplication implementation from [Fastor](https://github.com/romeric/Fastor) - a fast tensor algenra library
* `linear::para_strassen_multiply` is the parallelized version of strassen algorithm implementation which works very fast.
* `linear::matrixproduct` is the GEMM implementation that can be called directly on the data pointers of `linear::matrix` instances which can be accessed using `begin()` method.
* `linear::matmul_block` is blocked matrix multiplication. This performs better for some medium sized (64-256) matrices. Sometimes even better than `linear::matmul`.

Commands run in the cmd were:-
```bash
g++ main.cpp -o main -fopenmp -lblas -O3 -I./ -mfma -mavx
```
note: Fastor library was put in the same directory as the source file `main.cpp`, hence the `-I./` include flag.

The benchmarked times are averaged over 1000 iterations for double matrices of sizes 16-512. After that, the iterations are reduced to 50 and `linear::para_strassen_multiply` and `linear::matmul_block` are not averaged after 2048.

| Matrix Size (n) | linear::matmul | Fastor::matmul | linear::para_strassen_multiply | matrixproduct(GEMM) | linear::matmul_block |
|-------------|-----------------------|-----------------|----------------------| ------------------|------------|
|   16    |    0 ms   |   0 ms  |  0 ms   |   0.054 ms      |  0.074 ms
|  32    |    0 ms    |   0 ms    |    0.116 ms     |    0.025 ms     |  0.015 ms    
|  64    |    0.002 ms  |   0 ms    |    0.272 ms     |    0.053 ms     |  0.045 ms   
|  128    |    1.018 ms   |   0 ms     |   0.508 ms     |   0.099 ms    |  0.1 ms    
|   256   |     0.2 ms  |    0.298 ms   |    5.146 ms     |   0.292 ms    |  2.707 ms  
|512     |  3.835 ms    |  13.201 ms |  30.334 ms    | 2.854 ms | 24.004 ms
|1024    |  25.96 ms  |  - |  173.64 ms   | 27.68 ms | 192.1 ms
| 2048    |  184.98 ms  |  - | 1209 ms | 155.78 ms | 1675 ms
|  4096    | 961 ms |  - |  8155 ms | 855.22 ms | 13633 ms
| 8192 |  7025 ms |  - | - | 6833 ms | -



There are several other function calls for matrix multiplication in the library. Each has its own specialty, negatives and positives. There is a non-parallelized version of `linear::para_strassen_multiply`

## Documentation
 You can go to [Matrix wiki](https://github.com/DrakenWan/Matrix/wiki) to read documentation for example usage and reference.

## Updates
- (commit update timestamp: 030520240406). I have made MAJOR CHANGES! So I have removed the memory alignment because the code for that was not portable. I have added BLIS Framework inspired code (source material added as reference in the code file) for general matrix multiply. This has sped up the code significantly. There is also a standalone gemm function that works on double square matrices. All the code is haphazardly put as of now and I will make changes to make it more compact if possible. And all the benchmarks for matrix multiplication will be updated soon. As of now I am just committing the changes.
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
