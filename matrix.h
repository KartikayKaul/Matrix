#pragma once
#ifndef MATRIX_H
#define MATRIX_H

#include<iostream>
#include<omp.h> // for matrix multiplication speedup.
#include<stdexcept>
#include<fstream>
#include<random>
#include<limits>
#include<cmath>
#include<sstream>
#include<iomanip>
#include<complex>

namespace linear{
// macros for deallocation
#define deAlloc(x) delete[] x; x = NULL;

#define MATRIX_PRECISION 4

//path macros
#define SAVEPATH "matrixSaves/"
#define F_EXT ".trix"

//stringify variable name MACRO
#define CHANGE_ID_TO_STRING(x) (#x)


struct range {
  int start;
  int end;
  int length;
  
  range(int x) : start(0), end(x), length(end-start) {}
  range(int x, int y) : start(x), end(y), length(end-start) {} 
  const int size() {return end - start;}
};

//test code
//Defining type trait to check if a type is complex
template<typename DATA>
struct is_complex : std::false_type {};

template<typename DATA>
struct is_complex<std::complex<DATA>> : std::true_type {};
// test code ends

template<typename DATA>
class matrix {
    /*  
        DATA MEMEBERS:-
            val = flattened 2d array of type DATA in row major form
            row = number of rows of the matrix
            col = number of columns of the matrix
        Guideline:
            1: It is mandatory that you make use of data types that are of numerical
               type to initialise the internal values of matrix.
            2: It is recommended to use float or double numerical type to initialise
               the `val` array of the matrix object. Certain operations lose precision
                in results when using `int` numerical types 
            3: While using a matrix operation ensure you close the operands with 
               parenthesis with the binary or unary operator that is being used
                For example, to perform multiplication of A and B to give C we will
                             write "matrix<int> C = (A & B);"
    */
    static_assert(std::disjunction<std::is_arithmetic<DATA>, is_complex<DATA>>::value, "`matrix` class only supports only numerical types.");
    DATA *val;
    int row, col;

    //deallocate memory for Val
    void delMemoryforVal() {
        delete[] this->val;
        this->val = NULL;
    }

    // memory allocation for internal data structure holding the values
    void getMemoryforVal(int row, int col) {
        try {
            this->val = new DATA[row*col];
        }  catch(const std::bad_alloc& e) {
            std::cerr << "Heap memory allocation failed. "<< e.what()<<std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    // validate the Param values
    bool validateParams(int x, int y, int dim) {
        // explicitly used for `slice` operation
        bool validatex = (x > -1 && x < dim+1);
        bool validatey = (y > -1 && y < dim+1);
        bool validatexlty = (x < y);
        if( validatex && validatey && validatexlty)
            return true;
        else
            return false;
    }
    
    // validate the index
    bool isValidIndex(int r, int c) const {
        matrix<int> dims = this->getDims();
        int nRows = dims(0,0);
        int nCols = dims(0,1);
        return r >= 0 && r < nRows && c >= 0 && c < nCols;
    }

    /// Gaussian Elimination private method
    void gaussianElimination(matrix<DATA>&, int, int);

    /// Full Pivoting private method
    void pickPivotFullPivoting(int, int&, int&);

    public:
        // Getting matrix dimensions
        matrix<int> getDims() const {
            /*
                Returns a 1x2 integer matrix (say Dims) with row in Dims(0,0) and col in Dims(0,1) 
            */
           matrix<int> Dims(1,2);
           Dims(0,0) = this->row;
           Dims(0,1) = this->col;
           
           return Dims;
        }

        int rows() const {return this->row;}
        int cols() const {return this->col;}

        // initialize empty matrix
        matrix() {
            this->row = this->col = 0;
            /*
                following line of code is defining 0 size memory in
                heap. Weird thing I know. By theory it should 
                show undefined behaviour but this is an undefined matrix.
            */
            getMemoryforVal(this->row, this->col);
        }

        // initialize a square matrix
        matrix(int n) {
            this->row = this->col = n;
            getMemoryforVal(n,n);
        }

        // initialize a rectangular matrix
        matrix(int row, int col) {
            this->row = row;
            this->col = col;
            
            getMemoryforVal(this->row,this->col);
        }

        // initialize a square matrix using a flattened 2d array input
        matrix(DATA *data, int n) {
            this->row =
            this->col = n;
            getMemoryforVal(n,n);
            for(int i=0; i<n; i++)
                for(int j=0; j<n; j++)
                    *(val + i*n + j) = *(data + i*n + j);
        }

        // initialize a rectangle matrix using a flattened 2d array input
        matrix(DATA *data, int row, int col) {
            this->row = row;
            this->col = col;
            getMemoryforVal(this->row,this->col);
            for(int i=0; i<row; i++)
                for(int j=0; j<col; j++)
                    *(val + i*col + j) = *(data + i*col + j);
        }

        // initialize a row x col matrix with `value`
        matrix(int row, int col, DATA value) {
            this->row = row;
            this->col = col;
            getMemoryforVal(row,col);
            for(int i=0; i<row; i++)
                for(int j=0; j<col; j++)
                    *(val + i*col + j) = value;
        }

        // initialize using a 2d std::vector 
        matrix(std::vector<std::vector<DATA>> data) {
            this->row = data.size();
            this->col = data[0].size();
            getMemoryforVal(this->row, this->col);
            for(int i=0; i<this->row; i++)
                for(int j=0; j<this->col; j++)
                    *(val + i*(this->col) + j) = data[i][j];
        }

        //copy constructor
        matrix(const matrix<DATA> &m) {
            this->row = m.row;
            this->col = m.col;
            this->getMemoryforVal(this->row, this->col);
            for(int i=0; i<this->row; i++)
                for(int j=0; j<this->col; j++)
                    *(val + (this->col)*i + j) = *(m.val + i*m.col + j);
        }

        //initializer list
        matrix(std::initializer_list<std::initializer_list<DATA>> list) {
            if(list.size() == 0 || list.begin()->size() == 0)
                matrix();
            else {
                this->row = list.size();
                this->col = list.begin()->size();
                this->getMemoryforVal(this->row, this->col);
                
                int i=0, j=0;
                for(const auto& ROW : list) {
                    for(const auto& elem : ROW) {
                        *(val + (this->col)*i + j) = elem;
                        j++;
                    }
                    j=0;
                    i++;
                }
            }
        }

        // insert/update all the elements in row major form into the internal data structure
        void insertAll(int r=-1, int c=-1);

        // insert value at rth row and cth column
        void insertAt(DATA, int, int);

        // update using flattened array
        void updateWithArray(DATA*, int, int);

        // display contents in a 2d grid form
        void display(const std::string msg="Matrix:-");
        
        ~matrix() {
            delete[] val;
            val = NULL;
        }

        /////// MATRIX OPERATIONS
        matrix<DATA> &operator+=(matrix const& );
        matrix<DATA> &operator-=(matrix const& );
        matrix<DATA> &operator*=(const DATA);
    
        // Index operator
        DATA& operator()(int, int); //access an element of the matrix
        DATA operator()(int, int) const;

        //Assignment operator
        matrix<DATA> &operator=(const matrix<DATA>&);

        //change dimensions
        void changeDims(int r, int c) {
            /*
                This function will reset memory.
                It will reshape the matrix and
                remove old data and reallocate
                memory for new data.
                with caution.
                This is different than the reshape function.
                Reshape function does not delete the values 
                of original matrix. Infact it creates a new 
                matrix which it returns.
            */
           this->row = r;
           this->col = c;
           this->delMemoryforVal();
           this->getMemoryforVal(r, c);
        }
        
        // Reshape
        matrix<DATA> reshape(int newRow, int newCol);

        // Transpose operation
        matrix<DATA> operator!(); 
        matrix<DATA> transpose();
        matrix<DATA> T(){ return this->transpose();};

        /// Slice operation
        matrix<DATA> slice(int, int, int, int);
        matrix<DATA> operator()(range, range);

        // Element-wise exponent operation
        matrix<DATA> operator^(int);
        
        /// Swap Operations
        void swapRows(int,int);
        void swapCols(int, int);
        
        /// Generate Augmented matrix (vertical stack or horizontal stack)
        matrix<DATA> hStack(matrix const& ); // horizontal stack - hStack
        matrix<DATA> vStack(matrix const& ); // vertical  stack - vStack
        matrix<DATA> stack(matrix const& obj, bool vert=false); //generalized stack - stack

        /// aggregate functions
        matrix<DATA> max(int dim=-1);
        matrix<DATA> argmax(int dim=-1);
        matrix<DATA> min(int dim=-1);
        matrix<DATA> argmin(int dim=-1);

        /// matrix inverse operation
        matrix<DATA> inv(); //experimental

        // solve Ax = b
        matrix<DATA> solve(const matrix<DATA>&); //experimental

        // get determinant
        double det(bool fullPivot=false);
        double determinant(bool fullPivot=false) {
            return this->det(fullPivot);
        }

        /// QUERY methods
        bool isSquare() { if(this->col == this->row) return true; else return false;}
        bool isSymmetric();
        DATA item();
        bool isComparable(const matrix<DATA>&) const;

        /// FILE OPERATIONS I/O
        bool saveMatrix(const std::string&);
        bool loadMatrix(const std::string&);
};

//// NON-MEMBER OPERATIONS DECLARATIONS ///
template<typename DATA>
matrix<DATA> operator+(const matrix<DATA>&, const matrix<DATA>&);
template<typename DATA>
matrix<DATA> operator+(const matrix<DATA>&, const double);
template<typename DATA>
matrix<DATA> operator+(const double, const matrix<DATA>&);

template<typename DATA>
matrix<DATA> operator-(const matrix<DATA>&, const matrix<DATA>&);
template<typename DATA>
matrix<DATA> operator-(const matrix<DATA>&, const double);
template<typename DATA>
matrix<DATA> operator-(const double, const matrix<DATA>&);

template<typename DATA>
matrix<DATA> operator*(const matrix<DATA>&, const DATA);
template<typename DATA>
matrix<DATA> operator*(const DATA, const matrix<DATA>&);
template<typename DATA>
matrix<DATA> operator*(const matrix<DATA>&, const matrix<DATA>&);

template<typename DATA>
bool operator==(const matrix<DATA>&, const matrix<DATA>&);
template<typename DATA>
bool operator!=(const matrix<DATA>&, const matrix<DATA>&);

template<typename DATA>
matrix<DATA> operator/(const matrix<DATA>&, const double);
template<typename DATA>
matrix<DATA> operator/(const matrix<DATA>&, const matrix<DATA>&);

template<typename DATA>
matrix<DATA> operator&(const matrix<DATA>&, const matrix<DATA>&);

template<typename DATA>
matrix<DATA> eye(int);

template<typename DATA>
matrix<DATA> diagonal(int, DATA);

template<typename DATA>
bool is_triangular(matrix<DATA>&);

matrix<double> zeros(int);
matrix<double> zeros(int,int);
matrix<double> zeros_like(const matrix<double>);

matrix<double> randomUniform(int, double minVal=0., double maxVal=1.);
matrix<double> randomUniform(int, int, double minVal=0., double maxVal=1.);

matrix<int> randomUniformInt(int, int, int);
matrix<int> randomUniformInt(int, int, int, int);
/// Non-member operations declarations end ///

/// RESHAPE METHOD DEFINITION
template<typename DATA>
matrix<DATA> matrix<DATA>::reshape(int newRow, int newCol) {
    if(newRow * newCol != (this->cols())*(this->rows())){
        throw std::invalid_argument("The product of dimensions do not match the product of dimensions of the given matrix.");
    }

    matrix<DATA> reshapedMatrix(newRow, newCol);
    for(int i=0; i< ((this->cols()) * (this->rows())); i++ ) {
        reshapedMatrix(i/newCol, i%newCol) = val[i];
    }
    return reshapedMatrix;
}

/// Picking pivot using FULL PIVOTING 
template<typename DATA>
void matrix<DATA>::pickPivotFullPivoting(int startRow, int& pivotRow, int& pivotCol) {
    pivotRow = startRow;
    pivotCol = startRow;
    for(int i=startRow; i<this->rows(); i++) {
        for(int j=startRow; j<this->cols(); j++) {
            if( abs(val[i*(this->cols()) + j]) > abs(val[pivotRow * (this->cols()) + pivotCol]) ) {
                pivotRow = i;
                pivotCol = j;
            } 
        }
    }
}

/// SWAP OPERATIONS 
template<typename DATA>
void matrix<DATA>::swapRows(int row1, int row2) {
    if( !isValidIndex(row1, 0) || !isValidIndex(row2, 0)) {
        throw std::invalid_argument("Row dim indices are wrong.\n");
        return;
    }
    for(int j=0; j<col; j++) {
        DATA temp = *(val + row1*col + j);
        *(val + row1*col + j) = *(val + row2*col + j);
        *(val + row2*col + j) = temp;
    }
}
template<typename DATA>
void matrix<DATA>::swapCols(int col1, int col2) {
    if( !isValidIndex(0, col1) || !isValidIndex(0, col2)) {
        throw std::invalid_argument("Column dim indices are wrong.\n");
        return;
    }
    for (int i = 0; i<row; ++i) {
        DATA temp = *(val + i*col + col1);
        *(val + i*col + col1) = *(val + i*col + col2);
        *(val + i*col + col2) = temp;
    }
}

/// GAUSSIAN ELIMINATION DEFINITION
template<typename DATA>
void matrix<DATA>::gaussianElimination(matrix<DATA>& augMat, int n, int m) {
    //test
    int minDim = ((n < m) ? n : m);
    
     for(int i=0; i<minDim; i++) {  //test : minDim replaced n here
        //finding the pivot
        int pivotRow = i;
        for(int k=i+1; k<n; k++) {
            if(abs(augMat(k,i)) > abs(augMat(pivotRow, i))) {
                pivotRow = k;
            }
        }

        if(augMat(pivotRow,i) == 0) {
            throw std::domain_error("Matrix is singular. Cannot find inverse.");
        }

        //swapping rows in augMat
        augMat.swapRows(i, pivotRow);

        //Applying gaussian elimination
        DATA pivot = augMat(i, i);

        // row scaling
        for(int j=0; j< m; j++) {
            augMat(i, j) /=  pivot;
        }

        for(int k=0; k<n; k++) {
            if( k != i) {
                DATA factor = augMat(k, i);

                // row combine for augMat
                for(int j=0; j<m; j++) {
                    augMat(k, j) -= factor * augMat(i, j);
                }
            } // if k!= i condn end
        }
    }
}

//// SOLVE AX = B /////
template<typename DATA>
matrix<DATA> matrix<DATA>::solve(const matrix<DATA>& b) {
    int n = this->row;
    int m = this->col; // n x m dims

    if( n != m) {
        throw std::length_error("`solve` operation is only applicable to square matrices.");
    }

    // validate the shape of b
    if( b.rows() != n || b.cols() != 1) {
        throw std::invalid_argument("The dimensions do not match with A.");
    }
    
    matrix<DATA> augMat = this->hStack(b); // [A | b]
    gaussianElimination(augMat, augMat.rows(), augMat.cols());

    // get the solution from the rightmost colimns of augMat
    matrix<DATA> sol = augMat.slice(0, n, n, n+1);
    return sol;
}

/////// AGGREGATE FUNCTIONS ///////
///// Min
template<typename DATA>
matrix<DATA> matrix<DATA>::min(int dim) {
     DATA minElem;
    if(dim == -1) {
            /* Returns a 1x1 matrix of max value */
            matrix<DATA> m0(1,1);

            // might subroutine the below repetitive code
            minElem = *val;
            for(int i=1; i<this->row*this->col; i++)
                if( minElem > *(val + i)) {
                    minElem = *(val + i);
                }
            m0.insertAt(minElem, 0, 0);
            return m0;
    } //dim == -1
     else {
    
        if(dim == 0) {

                /* Returns a 1xcol matrix of max value in each jth column
                this operation is performed along the 0th axis (row axis)
                */
                matrix<DATA> m1(1, this->col);
                for(int j=0; j<this->col; j++) {
                    minElem = *(val +  j);

                    for(int i=1; i<this->row; i++)
                        if( minElem > *(val + i*(this->col) + j) )
                            minElem = *(val + i*(this->col) + j);
                    m1.insertAt(minElem, 0,j);
                }
                return m1;
        } // dim == 0

        else {
            if(dim == 1) {
                /*
                    Returns a rowx1 matrix of max value in each ith row
                    this operation is performed along the 1th axis (col axis)
                */
               matrix<DATA> m2(this->row, 1);
               for(int i=0; i<this->row; i++) {
                    minElem = *(val + i*(this->col));

                    for(int j=1; j<this->col; j++) {
                        if( minElem > *(val + i*(this->col) + j) )
                            minElem = *(val + i*(this->col) +j);
                    }
                    m2.insertAt(minElem, i, 0);
               }
              return m2;
            } else {
                throw std::invalid_argument("Invalid dim index used.");
            }
        } 
    }
}

//// Indices of Min element
template<typename DATA>
matrix<DATA> matrix<DATA>::argmin(int dim) {
int minIdx_i, minIdx_j;
    if(dim == -1) {
        /*
            Calculate the index of max element in entire matrix
            Returns a 1x2 matrix with ith index at (0,0) and jth index at (0,1)
        */
            matrix<DATA> m0(1,2);
            // might subroutine the below repetitive code
            minIdx_i = 0;
            minIdx_j = 0;
            DATA minElem = *(val + minIdx_i*(this->col) + minIdx_j);
            for(int i=0; i<this->row; i++) 
                for(int j=0; j<this->col; j++)
                    {
                        if(minElem > *(val + i*(this->col) + j))
                            {
                                minElem = *(val + i*(this->col) + j);
                                minIdx_i = i;
                                minIdx_j = j;
                            }
                    }
            // insert the indices at (0,0) and (0,1)
            m0.insertAt(minIdx_i, 0, 0);
            m0.insertAt(minIdx_j, 0, 1);
            return m0;
    } else {
        if(dim == 0) {

                /* Returns a 1xcol matrix of index of max value in each jth column
                this operation is performed along the 0th axis (row axis)
                */
                matrix<DATA> m1(1, this->col);
                for(int j=0; j<this->col; j++) {
                    int minIdx_i = 0;
                    DATA minElem = *(val + minIdx_i*(this->col) + j);
                    for(int i=1; i<this->row; i++)
                        if( minElem > *(val + i*(this->col) + j) ) {
                            minElem = *(val + i*(this->col) + j);
                            minIdx_i = i;
                        }
                    m1.insertAt(minIdx_i, 0,j);
                }
                return m1;
        } // dim == 0 condn end
         else {
            if(dim == 1) {
                /*
                    Returns a rowx1 matrix with index of max value in each ith row
                    this operation is performed along the 1th axis (col axis)
                */
               matrix<DATA> m2(this->row, 1);
               for(int i=0; i<this->row; i++) {
                    int minIdx_j = 0;
                    DATA minElem = *(val + i*(this->col) + minIdx_j);
                    
                    for(int j=1; j<this->col; j++) {
                        if( minElem > *(val + i*(this->col) + j) ) {
                            minElem = *(val + i*(this->col) + j);
                            minIdx_j = j;
                        } 
                    }
                    m2.insertAt(minIdx_j, i, 0);
               }
              return m2;
            } //dim == 1 condn end
            else {
                throw std::invalid_argument("The axis value is not correct.");
            }
        }
    }
}


//// Max
template<typename DATA>
matrix<DATA> matrix<DATA>::max(int dim) {
    DATA maxElem;
    if(dim == -1) {
            /* Returns a 1x1 matrix of max value */
            matrix<DATA> m0(1,1);

            // might subroutine the below repetitive code
            maxElem = *val;
            for(int i=1; i<this->row*this->col; i++)
                if( maxElem < *(val + i)) {
                    maxElem = *(val + i);
                }

            m0.insertAt(maxElem, 0, 0);
            return m0;
    } //dim == -1
     else {
    
        if(dim == 0) {

                /* Returns a 1xcol matrix of max value in each jth column
                this operation is performed along the 0th axis (row axis)
                */

                matrix<DATA> m1(1, this->col);

                for(int j=0; j<this->col; j++) {
                    maxElem = *(val +  j);

                    for(int i=1; i<this->row; i++)
                        if( maxElem < *(val + i*(this->col) + j) )
                            maxElem = *(val + i*(this->col) + j);

                    m1.insertAt(maxElem, 0,j);
                }
                return m1;
        } // dim == 0

        else {
            if(dim == 1) {
                /*
                    Returns a rowx1 matrix of max value in each ith row
                    this operation is performed along the 1th axis (col axis)
                */

               matrix<DATA> m2(this->row, 1);
               for(int i=0; i<this->row; i++) {
                    maxElem = *(val + i*(this->col));

                    for(int j=1; j<this->col; j++) {
                        if( maxElem < *(val + i*(this->col) + j) )
                            maxElem = *(val + i*(this->col) +j);
                    }

                    m2.insertAt(maxElem, i, 0);
               }
              return m2;
            } else {
                throw std::invalid_argument("Invalid dim index used.");
            }
        } 
    }
    
}

/// Indices of max
template<typename DATA>
matrix<DATA> matrix<DATA>::argmax(int dim) {
    int maxIdx_i, maxIdx_j;
    if(dim == -1) {
        /*
            Calculate the index of max element in entire matrix
            Returns a 1x2 matrix with ith index at (0,0) and jth index at (0,1)
        */
            matrix<DATA> m0(1,2);

            // might subroutine the below repetitive code
            maxIdx_i = 0;
            maxIdx_j = 0;
            DATA maxElem = *(val + maxIdx_i*(this->col) + maxIdx_j);
            for(int i=0; i<this->row; i++) 
                for(int j=0; j<this->col; j++)
                    {
                        if(maxElem < *(val + i*(this->col) + j))
                            {
                                maxElem = *(val + i*(this->col) + j);
                                maxIdx_i = i;
                                maxIdx_j = j;
                            }
                    }

            // insert the indices at (0,0) and (0,1)
            m0.insertAt(maxIdx_i, 0, 0);
            m0.insertAt(maxIdx_j, 0, 1);
            return m0;
    } else {
        if(dim == 0) {
                /* Returns a 1xcol matrix of index of max value in each jth column
                this operation is performed along the 0th axis (row axis)
                */
                matrix<DATA> m1(1, this->col);

                for(int j=0; j<this->col; j++) {
                    int maxIdx_i = 0;
                    DATA maxElem = *(val + maxIdx_i*(this->col) + j);
                    for(int i=1; i<this->row; i++)
                        if( maxElem < *(val + i*(this->col) + j) ) {
                            maxElem = *(val + i*(this->col) + j);
                            maxIdx_i = i;
                        }
                    m1.insertAt(maxIdx_i, 0,j);
                }
                return m1;
        } // dim == 0 condn end
         else {
            if(dim == 1) {
                /*
                    Returns a rowx1 matrix with index of max value in each ith row
                    this operation is performed along the 1th axis (col axis)
                */
               matrix<DATA> m2(this->row, 1);
               for(int i=0; i<this->row; i++) {
                    int maxIdx_j = 0;
                    DATA maxElem = *(val + i*(this->col) + maxIdx_j);
                    
                    for(int j=1; j<this->col; j++) {
                        if( maxElem < *(val + i*(this->col) + j) ) {
                            maxElem = *(val + i*(this->col) + j);
                            maxIdx_j = j;
                        } 
                    }
                    m2.insertAt(maxIdx_j, i, 0);
               }
              return m2;
            } //dim == 1 condn end
            else {
                throw std::invalid_argument("The axis value is not correct.");
            }
        }
    }
}

/// Assignment overload
template<typename DATA>
matrix<DATA>& matrix<DATA>::operator=(const matrix<DATA>& m1) {
    this->updateWithArray(m1.val, m1.rows(), m1.cols());
    return *this;
}

/// COMPARE DIMENSIONS
template<typename DATA>
bool matrix<DATA>::isComparable(const matrix<DATA>& m) const {
    if(this->rows() == m.rows() && this->cols() == m.cols()) {
        return true;
    }
    return false;
}

template<typename DATA>
bool matrix<DATA>::isSymmetric() {
    if(this->row == this->col)
     {
        matrix<DATA> Transpose = this->T();
        if(Transpose == *this)
            return true;
     }
     return false;
}

template<typename DATA>
DATA matrix<DATA>::item() {
    if(this->row == 1  && this->col == 1) {
        return *val; 
    } else {
        throw std::invalid_argument("To throw an item out it is supposed to be 1x1 matrix.");
    }
}

/// STACKING OPERATIONS
template<typename DATA>
matrix<DATA> matrix<DATA>::stack(matrix const& obj, bool vert) {
    if(vert)
        return this->vStack(obj);
    else
        return this->hStack(obj);
}

template<typename DATA>
matrix<DATA> matrix<DATA>::hStack(matrix const& obj) {
    if(this->row != obj.row)
        throw std::invalid_argument("The row dimensions do not match.");
    
    // initialize the augmented matrix
    matrix<DATA> m(this->row, this->col + obj.col);
    for(int i=0; i<m.row; i++) {
        for(int j=0; j<this->col; j++)
            *(m.val + i*m.col + j) = *(val + (this->col)*i + j);

        for(int j=0; j<obj.col; j++)
            *(m.val + i*m.col + (j+this->col)) = *(obj.val + i*obj.col + j);
    }
    return m;
}

template<typename DATA>
matrix<DATA> matrix<DATA>::vStack(matrix const& obj) {
    if(this->col != obj.col)
        throw std::invalid_argument("The column dimensions do not match.");

    // initialize our augmented matrix
    matrix<DATA> m(this->row + obj.row, this->col);
    for(int j=0; j<m.col; j++) {
        for(int i=0; i<this->row; i++)
            *(m.val + i*m.col + j) = *(val + i*m.col + j);

        for(int i=0; i<obj.row; i++)
            *(m.val + (i+this->row)*m.col + j) = *(obj.val + i*obj.col + j);
    }
    return m;
}

template<typename DATA>
matrix<DATA> matrix<DATA>::inv() {
    int n = this->row;
    int m = this->col;

    if(n != m) {
        std::length_error("Inverse cannot be calculated for non-square matrices.");
    }

    matrix<DATA> I = eye<DATA>(n); // nxn Identity matrix
    matrix<DATA> augMat = this->hStack(I); // nx2n augmented Matrix

    // Calling gaussianElimination on augMat
    this->gaussianElimination(augMat, augMat.rows(), augMat.cols());

    // inverse present in right half of augmented matrix (augMat)
    matrix<DATA> inverse = augMat.slice(0, n, n, 2*n);
    return inverse;
}

// DETERMINANT
template<typename DATA>
double matrix<DATA>::det(bool fullPivot) {
    // check that matrix is square
    if(!isSquare()) {
        throw std::domain_error("Determinant not defined for non-square matrices.");
    }

    // get dimensions
    matrix<int> dims = this->getDims();
    int n = dims(0,0); //rows
    int m = dims(0,1); //cols
    
    matrix<DATA> this_copy(this->val, n, m); //copy of this
    double detValue = 1.0;
    double sign = 1;

    if(fullPivot) {
        for(int i=0; i<n-1; i++) {
            // find pivot and perform full pivoting
            int pivotRow, pivotCol;
            this_copy.pickPivotFullPivoting(i, pivotRow, pivotCol);

            // swap rows and cols
            if(pivotRow != i) {
                this_copy.swapRows(i, pivotRow);
                sign *= -1; //row swap causes sign change
            }

            if(pivotCol != i) {
                this_copy.swapCols(i, pivotCol);
                sign *= -1; // col swap causes sign change
            }

            double pivot = this_copy(i,i);
            if(abs(pivot) == 0.) {
                return 0.; // singular!
            }       

            for(int k=i+1; k<n; k++) {
                double factor = this_copy(k,i) / pivot;
                
                for(int j=i; j<n; j++) {
                    this_copy(k, j) -= factor * this_copy(i, j);
                }
            }
            detValue *= pivot; //multiply detValue by pivot value 
    }
    detValue *= this_copy(n-1, n-1);
    detValue *= sign;     //finally implement the sign into it
    } else { //partial pivoting

        for(int i=0; i< n-1; i++) {
            int pivotIdx=i;
            double maxVal = abs(this_copy(i,i));
            for(int k=i+1; k<n; k++) {
                double val = abs(this_copy(k,i));
                if(val > maxVal)
                {
                    pivotIdx = k;
                    maxVal = val;
                }
            }
            if(pivotIdx != i) {
                this_copy.swapRows(i,pivotIdx);
                sign *= -1;
            }

            // row operations that will turn it to UTM
            double pivot = this_copy(i,i);
            if(abs(pivot) == 0.)
                return 0.;
            
            for(int k=i+1; k<n; k++) {
                double factor = this_copy(k,i) / pivot;
                for(int j=i; j<n; j++)
                    this_copy(k,j) -= factor * this_copy(i,j);
            }
            detValue *= pivot;
        }
        detValue *= this_copy(n-1,n-1);
        detValue *= sign;
    } //partial pivoting ends here
    return detValue;
}

/// ELEMENT-WISE EXPONENT OPERATION
template<typename DATA>
matrix<DATA> matrix<DATA>::operator^(int power) {
    matrix<DATA> m(this->row, this->col);

    for(int i=0; i<m.rows(); i++) {
        for(int j=0; j<m.cols(); j++) {
            DATA prod=1.;
            prod = pow(*(val + i*(this->col) + j), power);
            m(i,j) = prod;
        }
    }
    return m;
}   

/// INDEX OPERATION
template<typename DATA>
DATA& matrix<DATA>::operator()(int r, int c)  {
    
    if(r < this->row && c < this->col) {
        return *(val + r*this->col + c);
    } else {
        throw std::invalid_argument("Indices exceed the dimension size.");
    }
}
template<typename DATA>
DATA matrix<DATA>::operator()(int r, int c) const {
    if(r < this->row && c < this->col) {
        return *(val + r*this->col + c);
    } else {
        throw std::invalid_argument("Indices exceed the dimension size.");
    }

}

/// SLICE OPERATION
template<typename DATA>
matrix<DATA> matrix<DATA>::operator()(range rowRange, range colRange) {
    return this->slice(rowRange.start, rowRange.end, colRange.start, colRange.end);
}
template<typename DATA>
matrix<DATA> matrix<DATA>::slice(int x_0, int y_0, int x_1, int y_1) {
    /*
        Takes in 4 integer parameters:-
            x_0 : start index of row dimension
            y_0 : end index of row dimension (exclusive)
            x_1 : start index of col dimension
            y_1 : end index of col dimension (exclusive)

        RETURNS 
           a (y_0 - x_0) x (y_1 - x_1) submatrix from the given input matrix
        
        USAGE
            Suppose you have a 5x4  matrix of some random integervalues
            You invoke `matrix<int> subMatrix = A.slice(1,4, 1,2)`. This will
            return a 3x1 submatrix of the original matrix.

              5x4 Matrix                        3x1 sub matrix
            [[1, 2, 3, 4],                          
             [1, 4, 3, 2],                           [[4],
             [3, 1, 2, 4],         =====>             [1],
             [2, 3, 4, 1],                            [3]] 
             [4, 2, 1, 3]]
    */
    bool validation = (this->validateParams(x_0, y_0, this->row)) && (this->validateParams(x_1, y_1, this->col));
    //std::cout<<"Params _0"<<x_0<<','<<y_0;
    //std::cout<<"\nParams _1"<<x_1<<','<<y_1;

    if (validation) {
        matrix<DATA> m(y_0 - x_0, y_1 - x_1);

        for(int i=0; i<m.row; i++) {
            for(int j=0; j<m.col; j++) {
                *(m.val + i*m.col + j) = *(val + (i + x_0)*(this->col) + (j + x_1));
            }
        }
        return m;
    } else {
        throw std::invalid_argument("Wrong index range received. Check your index Params.");
    }
}

/// TRANSPOSE OPERATION
template<typename DATA>
matrix<DATA> matrix<DATA>::operator!() {
    matrix<DATA> m(this->col, this->row);

    //using insertAt operation to fill in the elements
    for(int i=0; i<m.row; i++)
        for(int j=0; j<m.col; j++)
            m.insertAt(*(val + (this->col)*j + i),i,j);
    return m;
}

/// TRANSPOSE OPERATION
template<typename DATA>
matrix<DATA> matrix<DATA>::transpose() {
    matrix m = !(*this);
    return m;
}

/// Insertion Operations
template<typename DATA>
void matrix<DATA>::insertAll(int r, int c)  {
    if(r > -1 &&  c > -1)
        this->changeDims(r, c);
    int i,j;
    std::cout<<"\nNote: you have to insert "<<this->row*this->col<<" values. Values will be filled row-major wise in a "<<this->row<<'x'<<this->col<<" matrix.\n";
    for(i=0; i<this->row; i++)
        for(j=0; j<this->col; j++)
            std::cin>>*(val + (this->col)*i + j);
}
template<typename DATA>
void matrix<DATA>::insertAt(DATA value, int r, int c)  {
        if( (r>-1 && r < this->row) && (c>-1 && c<this->col)) {
            *(val + (this->col)*r + c) = value;
        } else {
            throw std::invalid_argument("The index values exceed the dimension size of the matrix.");
        }
}
template<typename DATA>
void matrix<DATA>::updateWithArray(DATA* array, int r, int c) {
    if (r <0 || c < 0)
        throw std::invalid_argument("Bad dimension values.");

    this->changeDims(r, c);
    for(int i=0; i<this->row; i++)
        for(int j=0; j<this->col; j++)
            *(val + i*(this->col) + j) = *(array + i*(this->col) + j);
}

/// Print matrix in ostream
template<typename DATA>
void matrix<DATA>::display(const std::string msg)  {
    //experimental code
    int i,j;
    std::cout<<'\n'<<msg<<'\n';

    // zero size matrix display
    if(this->row == 0 || this->col == 0) {
        std::cout<<"(empty matrix)\n";
        return;
    }

    int max_precision = MATRIX_PRECISION;
    int padding = 1;

    // Find the maximum number of digits in the matrix
    int maxDigits = 1;
    for (i = 0; i < this->row; i++) {
        for (j = 0; j < this->col; j++) {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(max_precision) << *(val + (this->col) * i + j);
            std::string str = stream.str();

            size_t pos = str.find_last_not_of('0');
            if (pos != std::string::npos && str[pos] == '.')
                pos--;

            maxDigits = std::max(maxDigits, static_cast<int>(pos + 1));
        }
    }

    // Set the width based on the maximum number of digits
    int width = maxDigits + padding;
    for (i = 0; i < this->row; i++) {
        for (j = 0; j < this->col; j++) {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(max_precision) << *(val + (this->col) * i + j);
            std::string str = stream.str();

            size_t pos = str.find_last_not_of('0');
            if (pos != std::string::npos && str[pos] == '.')
                pos--;
            std::cout << std::setw(width) << str.substr(0, pos + 1);
        }
        std::cout << "\n";
    }
}

/// File operation on saving a matrix
template<typename DATA>
bool matrix<DATA>::saveMatrix(const std::string& filename) {
    std::string fullpath = SAVEPATH + filename + F_EXT;
    std::ofstream saveFile(fullpath);

    if(saveFile.is_open()) {
        // saving the dimensions
        saveFile<<row<<" "<<col<<"\n";

        // saving the flattened array elements (row major)
        for(int i=0; i<row; i++) {
            for(int j=0; j<col; j++) {
                saveFile<<*(val + i*col + j)<<" ";
            }
            saveFile<<"\n";
        }

        saveFile.close();
        std::cout<<"Matrix saved as file `"<<filename<<F_EXT<<"` successfully.\n";
        return true;
    } else {
        std::cout<<"Unable to open file at  `"<<fullpath<<"`\n";
        return false;
    }
}

/// File operation on loading a matrix
template<typename DATA>
bool matrix<DATA>::loadMatrix(const std::string& filename) {
    std::string fullpath = SAVEPATH + filename + F_EXT;
    std::ifstream loadFile(fullpath);

    if(loadFile.is_open()) {
        int loadR, loadC;
        loadFile>>loadR>>loadC;

        //resize the matrix
        this->changeDims(loadR, loadC);

        //reading matrix values from file
        for(int i=0; i<this->row; i++){
            for(int j=0; j<this->col; j++) {
                loadFile >> *(val + i*(this->col) + j);
            }
        }

        loadFile.close();
        std::cout<<"Matrix has been successfully loaded from `"<<fullpath<<"`.\n";
        return true;
    } else {
        return false;
    }
}

// Diagonal Matrix generator : one single value
template<typename DATA>
matrix<DATA> diagonal(int n, DATA value) {
    matrix<DATA> m(n);
    
    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++)
            {
                if(i==j)
                    m.insertAt(value, i, j);
                else
                    m.insertAt(0, i, j);
            }
    return m;
}

// Identity matrix of size n
template<typename DATA>
matrix<DATA> eye(int n) {
    matrix<DATA> m(n);
    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++) {
            if(i == j) {
                m.insertAt(1, i, j);
            } else {
                m.insertAt(0, i, j);
            }
        }
    return m;
}

// Is it triangular?
template<typename DATA>
bool is_triangular(matrix<DATA>& M) {
    if(!M.isSquare()) {
        throw std::invalid_argument("The matrix is not square.");
    }
    matrix<int> dims = M.getDims();
    int n = dims(0, 0);
    int m = dims(0, 1);

    // machine epsilon
    DATA epsilon = std::numeric_limits<DATA>::epsilon();
    bool upper=true, lower=true;

    //check upper triangular
    for(int i=1; i<n; i++) {
        for(int j=0; j<i && j<m; j++) {
            if(std::abs(M(i,j)) > epsilon) {
                upper = false;
                break;
            }
        }
    } // upper triangular

    //check lower triangular
    for(int i=0; i<n; i++) {
        for(int j=i+1; j<m; j++) {
            if( std::abs(M(i,j)) > epsilon ) {
                lower = false;
                break;
            }
        }
    } // lower triangular
    return (upper || lower);
}

// zero matrix square
matrix<double> zeros(int n) {
    matrix<double> _0s(n,n,0);
    return _0s;
}

// zero matrix rectangle/square
matrix<double> zeros(int n, int m) {
    matrix<double> _0s(n,m,0);
    return _0s;
}

// zero matrix like another matrix
matrix<double> zeros_like(const matrix<double> m) {
    matrix<double> _0s(m.rows(), m.cols(), 0);
    return _0s;
}

// random square matrix
matrix<double> randomUniform(int n, double minVal, double maxVal) {
    matrix<double> mat(n);

    std::random_device dev;
    std::mt19937 generator(dev());
    std::uniform_real_distribution<double> distribution(minVal, maxVal);

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++)
            mat(i,j) = distribution(generator);
    }
    return mat;
}

// random nxm matrix
matrix<double> randomUniform(int n, int m, double minVal, double maxVal) {
    matrix<double> mat(n,m);

    std::random_device dev;
    std::mt19937 generator(dev());
    std::uniform_real_distribution<double> distribution(minVal, maxVal);

    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++)
            mat(i,j) = distribution(generator);
    }
    return mat;
}

// random integer square matrix
matrix<int> randomUniformInt(int n, int minVal, int maxVal) {
    matrix<int> mat(n);

    std::random_device dev;
    std::mt19937 generator(dev());
    std::uniform_int_distribution<int> distribution(minVal, maxVal);

    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++)
            mat(i,j) = distribution(generator);
    }

    return mat;
}

// random integer nxm matrix
matrix<int> randomUniformInt(int n, int m, int minVal, int maxVal) {
    matrix<int> mat(n,m);

    std::random_device dev;
    std::mt19937 generator(dev());
    std::uniform_int_distribution<int> distribution(minVal, maxVal);

    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++)
            mat(i,j) = distribution(generator);
    }

    return mat;
}

template<typename DATA>
matrix<DATA> &matrix<DATA>::operator*=(const DATA value) {
    for(int i=0; i < row*col; i++)
        *(val + i) *= value;
    return *this;
}

template<typename DATA>
matrix<DATA> &matrix<DATA>::operator-=(const matrix<DATA>& m1) {
    if(!this->isComparable(m1))
        throw std::invalid_argument("Dimensions do not match.");
    else {
        for(int i=0; i<this->rows()*this->cols(); i++)
            *(val + i) -= m1(i/m1.cols(), i%m1.cols());
    }
    return *this;
}

template<typename DATA>
matrix<DATA> operator*(const matrix<DATA>& m1, const DATA value) {
    matrix<DATA> result = m1;
    result *= value;
    return result;
}
template<typename DATA>
matrix<DATA> operator*(const DATA val, const matrix<DATA>& m1) {
    return m1 * val;
}
template<typename DATA>
matrix<DATA> operator*(const matrix<DATA>& m1, const matrix<DATA>& m2) {
    // element-wise multiplication 
    // NOT MATRIX MULTIPLICATION
    matrix<DATA> product(m1.rows(), m1.cols(), (DATA)1);
    for(int i=0; i<m1.rows()*m1.cols(); i++){
        product(i/m1.cols(), i%m1.cols()) *= (m1(i/m1.cols(), i%m1.cols()) * m2(i/m2.cols(), i%m2.cols()));
    }
    return product;
}

template<typename DATA>
bool operator==(const matrix<DATA>& m1, const matrix<DATA>& m2) {
    if(m1.isComparable(m2)) {
        bool equal = true;
        int n = m1.rows();
        int m = m1.cols();
        for(int i=0; i<n*m; i++)
            {
                if(m1(i/m, i%m) != m2(i/m, i%m))
                    {
                        equal = false;
                        break;
                    }
            }
        return equal;
    } else {
        return false;
    }
}

template<typename DATA>
matrix<DATA> &matrix<DATA>::operator+=(const matrix<DATA>& m1) {
    if(!this->isComparable(m1))
        throw std::invalid_argument("Dimensions do not match.");
    else {
        for(int i=0; i<this->rows()*this->cols(); i++)
            *(val + i) += m1(i/m1.cols(), i%m1.cols());
    }
    return *this;
}

template<typename DATA>
matrix<DATA> operator+(const matrix<DATA>& m1, const matrix<DATA>& m2) {
   matrix<DATA> m = m1;
   m += m2;
   return m;
}
template<typename DATA>
matrix<DATA> operator+(const matrix<DATA>& m1, const double value) {
    int Size = m1.rows() * m1.cols();
    for(int i=0; i<Size; i++)
        m1(i/m1.cols(), i%m1.cols()) += value;
    return m1;
}
template<typename DATA>
matrix<DATA> operator+(const double value, const matrix<DATA>& m2) {
    return m2+value;
}

template<typename DATA>
matrix<DATA> operator-(const matrix<DATA>& m1, const matrix<DATA>& m2){
    matrix<DATA> m = m1;
    m += m2;
    return m;
}
template<typename DATA>
matrix<DATA> operator-(const matrix<DATA>& m1, const double value) {
    int Size = m1.rows() * m1.cols();
     for(int i=0; i<Size; i++)
        m1(i/m1.cols(), i%m1.cols()) -= value;
    return m1;
}
template<typename DATA>
matrix<DATA> operator-(const double value, const matrix<DATA>& m2) {
    return m2-value;
}

/// MATRIX MULTIPLICATION
template<typename DATA>
matrix<DATA> operator&(const matrix<DATA> &m1,const matrix<DATA> &m2) {
    if(m1.cols() != m2.rows()) {
        throw std::invalid_argument("Internal dimensions do not match.");   
    }
    
    int i, j, k;
    matrix<DATA> m(m1.rows(), m2.cols(), (DATA)0);
    if (m1.rows() >= 100 || m1.cols() >= 100|| m2.cols() >= 100) {
        matrix<DATA> m(m1.rows(), m2.cols(), (DATA)0);
        #pragma omp parallel for private(i,j,k) shared(m1, m2, m)
        for(i=0; i<m1.rows(); i++)
            for(k=0; k<m2.rows(); k++)
                for(j=0; j<m2.cols(); j++)
                    m(i,j) += m1(i,k) * m2(k,j);
        return m;
    }
    else {
        for(i=0; i<m1.rows(); i++)
            for(k=0; k<m2.rows(); k++)
                for(j=0; j<m2.cols(); j++)
                    m(i,j) += m1(i,k) * m2(k,j);
        return m;
    } 
}
} //linear namespace

template<typename DATA>
void init2dArray(DATA *array, int size_0, int size_1) {
    /*
        UTIL FUNCTION
        Flattened 2d array in row major form will be initialised using this
    */
   std::cout<<"\nPlease insert "<<size_0*size_1<<" values in row major form for a "<<size_0<<'x'<<size_1<<" matrix:-\n";
    for(int i=0; i<size_0; i++)
        for(int j=0; j<size_1; j++)
            std::cin>>*(array + i*size_1 + j);
}

void init2dRandArray(int *array, int size_0, int size_1, int start=0, int end=9) {
    /*
     UTIL FUNCTION
        Flattened 2d array in row major form will be initialised using a
        uniform integer distribution.
    */
    for (int i = 0; i < size_0; i++) {
        std::default_random_engine generator(std::random_device{}());
        std::uniform_int_distribution<int> distribution(start, end);
        for (int j = 0; j < size_1; j++)
            *(array + i * size_1 + j) = distribution(generator);
    }
}

void init2dRandArray(float *array, int size_0, int size_1, float start=0., float end=1.) {
    
    for (int i = 0; i < size_0; i++) {
        std::default_random_engine generator(std::random_device{}());
        std::uniform_real_distribution<float> distribution(start, end);
        for (int j = 0; j < size_1; j++)
            *(array + i * size_1 + j) = distribution(generator);
    }
}

void init2dRandArray(double *array, int size_0, int size_1, double start=0., double end=1.) {
    
    for (int i = 0; i < size_0; i++) {
        std::default_random_engine generator(std::random_device{}());
        std::uniform_real_distribution<double> distribution(start, end);
        for (int j = 0; j < size_1; j++)
            *(array + i * size_1 + j) = distribution(generator);
    }
}

void init2dRandArray(std::complex<double> *array, int size_0, int size_1, double start=-1., double end=1.) {

    std::complex<double> randomComplexNumber;
    for (int i = 0; i < size_0; i++) {
        std::default_random_engine generator(std::random_device{}());
        std::uniform_real_distribution<double> realDisbn(start, end);
        std::uniform_real_distribution<double> imagDisbn(start, end);
        for (int j = 0; j < size_1; j++) {
                double realPart = realDisbn(generator);
                double imagPart = imagDisbn(generator);
                randomComplexNumber.real(realPart);
                randomComplexNumber.imag(imagPart);
                 *(array + i * size_1 + j) = randomComplexNumber;
        }          
    }
}
#endif // MATRIX_H