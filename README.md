# Matrix
 Linear Algebra library in C++

The matrix code as of now is just a copy of the matrix data structure I made in [AbstractDS repo](https://github.com/DrakenWan/Abstract-Data-Structures). I am going to now create a standalone linear algebra library in c++. This is just me experimenting for my scientific computing course.



## Updates
- (timestamp: 1401240408) `reshape` method has been implemented with correct logic. Working fine. Will keep testing for boundary and special cases.
- Currently working on `determinant()` method. Even though this is the most useless linear algebra function but still I am working on it. I will try to create LU decomposition function and try to calculate the determinant using `det(A) = det(L) * det(U) if A = LU`.