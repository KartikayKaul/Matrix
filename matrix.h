#pragma once
#ifndef MATRIX_H
#define MATRIX_H

#include<iostream>
#include<cstdlib>
#include<omp.h>
#include<stdexcept>
#include<fstream>
#include<random>
#include<limits>
#include<cmath>
#include<sstream>
#include<iomanip>
#include<complex>
#include<type_traits>
#include<x86intrin.h>

// Check if g++ or clang used except INTEL
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#define GCCAVAIL 1
#else
#define GCCAVAIL 0
#endif

// safeguard to check if cblas can be added
#ifdef __cplusplus
extern "C" {
#endif
#if GCCAVAIL
#include <cblas.h>
#endif
#ifdef __cplusplus
}
#endif

//testing
#include<algorithm>

namespace linear{
// macro for deallocation
#define deAlloc(x) delete[] x; x = NULL;

// precision macros
#define MATRIX_PRECISION 4 // precision of matrix values in console display
#define MATRIX_PRECISION_TOL 6 
#define PRECISION_TOL(type) std::numeric_limits<type>::epsilon() * std::pow(10, MATRIX_PRECISION_TOL)

//path macros for IO
#define SAVEPATH "saves" //default path
#define F_EXT ".linmat" //default linear::matrix format

//stringify variable name MACRO
#define CHANGE_ID_TO_STRING(x) (#x)

// linear::range is specially made for slicing operation to generate submatrices
struct range {
  int start;
  int end;
  int length;
  
  range(int x) : start(0), end(x), length(end-start) {}
  range(int x, int y) : start(x), end(y), length(end-start) {} 
  const int size() {return end - start;}
};

//Defining type trait to check if a type is std::complex
template<typename DATA>
struct is_complex : std::false_type {};
template<typename DATA>
struct is_complex<std::complex<DATA>> : std::true_type {};

template<typename DATA> //whether it is qualified to be a matrix value
constexpr bool is_numeric_v = std::is_arithmetic_v<DATA> || is_complex<DATA>::value;

// This has been added to be able to print the data type of the matrix
template <typename T>
struct TypeName { //defaulting case if i missed any numerical type
    static constexpr const char* value = "unknown";
};

#define DEFINE_TYPENAME_FOR_TYPE(TYPE)\
    template <>\
    struct TypeName<TYPE> {\
        static constexpr const char* value = #TYPE;\
    };

DEFINE_TYPENAME_FOR_TYPE(bool)
DEFINE_TYPENAME_FOR_TYPE(char)
DEFINE_TYPENAME_FOR_TYPE(signed char)
DEFINE_TYPENAME_FOR_TYPE(unsigned char)
DEFINE_TYPENAME_FOR_TYPE(short)
DEFINE_TYPENAME_FOR_TYPE(unsigned short)
DEFINE_TYPENAME_FOR_TYPE(int)
DEFINE_TYPENAME_FOR_TYPE(unsigned int)
DEFINE_TYPENAME_FOR_TYPE(long)
DEFINE_TYPENAME_FOR_TYPE(unsigned long)
DEFINE_TYPENAME_FOR_TYPE(long long)
DEFINE_TYPENAME_FOR_TYPE(unsigned long long)
DEFINE_TYPENAME_FOR_TYPE(float)
DEFINE_TYPENAME_FOR_TYPE(double)
DEFINE_TYPENAME_FOR_TYPE(long double)
DEFINE_TYPENAME_FOR_TYPE(std::complex<double>)
DEFINE_TYPENAME_FOR_TYPE(std::complex<int>)
DEFINE_TYPENAME_FOR_TYPE(std::complex<float>)
DEFINE_TYPENAME_FOR_TYPE(std::complex<long double>)

template<typename DATA=double>
class matrix {
    /*  
        DATA MEMBERS:- (- private; + public)
           - val = flattened 2d array of type DATA in row major form
           - row = number of rows of the matrix
           - col = number of columns of the matrix
           - *first = points to the first location of the *val
           - *last = points to the last value in *val (*last <- *val + (row * col))
    */
    static_assert(is_numeric_v<DATA>, "`linear::matrix` class only supports numerical types.");
    DATA *val;
    int row, col;
    DATA *first = NULL;
    DATA *last = NULL;

    // itrs for chained-initialization (check operator<<)
    int cur_row=0, cur_col=0;

    //deallocate memory for Val
    void delMemoryforVal() {
        if (this->rows() > 0 && this->cols() > 0) {
            delete[] this->val;
            this->val = NULL;
        }
    }

    // memory allocation for internal data structure holding the values
    void getMemoryforVal(int r, int c) {
        if(r<1 || c<1)
            throw std::invalid_argument("getMemoryforVal() - invalid dimension values.");
        try {
            val = new DATA[r*c];
        } catch (const std::bad_alloc& e) {
            std::cerr << "Heap memory allocation failed: " << e.what() << "\n";
            throw;
        }
        this->row = r;
        this->col = c;

        //experimental
        first = this->val;
        last = this->val + (r*c);
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
        int nRows = this->rows();
        int nCols = this->cols();
        return r >= 0 && r < nRows && c >= 0 && c < nCols;
    }

    /// Gaussian Elimination private method
    void gaussianElimination(matrix<DATA>&, int, int);

    /// Full Pivoting private method
    void pickPivotFullPivoting(int, int&, int&);
    
    // declaring class of different types as friend
    template<typename ATAD>
    friend class matrix;
    /* enable private member access within the set
    of matrix class template*/

    public:
         //// EXPERIMENTAL FUNCTIONS ////
         //print the type of matrix in console
         void print_type() {
            std::cout<<'\n'<<TypeName<DATA>::value<<'\n';
         }

         //returns string of the type of the matrix
         static constexpr  const char* type_s() {
            return TypeName<DATA>::value;
         }

         //First and Last accessors
         DATA* begin() const {
            return this->first;
         }
         DATA* end() const {
            return this->last;
         }

        // get total memory of the data
        size_t getTotalMemory() const {
            return sizeof(DATA) * static_cast<size_t>(last - first);
        }

        //swap matrices contents
        void swapValues(matrix<DATA>& other) {
            if(other.rows() != this->rows() || other.cols() != this->cols()) {
                throw std::domain_error("swapValues() - matrices have different dimensions.\n");
            }
            std::swap_ranges(this->first, this->last, other.first);
        }

        void fillUpperTriangle(DATA value)
         {
            int Rows = this->rows();
            int Cols = this->cols();

            for(DATA *i = this->first; i!= this->last; ++i) {
                int rowIdx = (i - first) / Cols;
                int colIdx = (i - first) % Cols;
                if(colIdx > rowIdx)
                    *i = value;
            }
         }
         void fillTriu(DATA value) {
            this->fillUpperTriangle(value);
         }
         void fillLowerTriangle(DATA value) {
            int Rows = this->rows();
            int Cols = this->cols();

            for(DATA *i = this->first; i!= this->last; ++i) {
                int rowIdx = (i - first) / Cols;
                int colIdx = (i - first) % Cols;
                if(colIdx < rowIdx)
                    *i = value;
            }
         }
         void fillTril(DATA value) {
            this->fillLowerTriangle(value);
         }
        //// EXPERIMENTAL ENDS ////

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

        //change dimensions
        void changeDims(int r, int c) {
            /*
                This function will reset memory.
                It will reshape the matrix and
                remove old data and reallocate
                memory for new data.
                This is different than the reshape function.
                Reshape function does not delete the values 
                of original matrix. Infact it creates a new 
                matrix which it returns.
            */
           this->delMemoryforVal();
           this->getMemoryforVal(r, c);
        }

        // initialize empty matrix, trivial
        matrix() {
            this->row = this->col = 0;
            this->first = this->last = NULL;
        }

        // initialize a square matrix, semi-init (lifetime hasn't started yet)
        matrix(int n) {
            getMemoryforVal(n,n);
        }

        // initialize a rectangular matrix, semi-init (lifetime hasn't started yet)
        matrix(int row, int col) {
            getMemoryforVal(row,col);
        }

        // initialize a square matrix using a flattened 2d array input
        matrix(DATA *data, int n) {
            getMemoryforVal(n,n);
            std::copy(data, data + n*n, this->first);
        }

        // initialize a rectangle matrix using a flattened 2d array input
        matrix(DATA *data, int row, int col) {
            getMemoryforVal(row,col);
            std::copy(data, data + row*col, this->first);
        }

        // initialize a row x col matrix with `value`
        matrix(int row, int col, DATA value) {
            getMemoryforVal(row,col);
            std::fill(this->first, this->last, value);
        }
        
        matrix(int row, int col, std::complex<DATA> value) {
            getMemoryforVal(row,col);
            std::fill(this->first, this->last, value);
        }

        // initialize using a 2d std::vector 
        matrix(std::vector<std::vector<DATA>> data) {
            getMemoryforVal(data.size(), data[0].size());

            int stride = 0;
            #pragma omp parallel for if(this->rows() >= 100)
            for(int i=0; i<this->rows(); ++i) {
                std::copy(data[i].begin(), data[i].end(), val + stride);
                stride += this->cols();
            }
        }

        //initialize a row vector 1xn using 1d std::vector
        matrix(std::vector<DATA> data) {
            getMemoryforVal(1, data.size());
            std::copy(data.begin(), data.end(), this->first);
        }

        //copy constructor
        matrix(const matrix<DATA> &m) {
            this->getMemoryforVal( m.rows(), m.cols());
            std::copy(m.begin(), m.end(), this->val);
        }
        //copy constructor for handling different type matrices
        template<typename ATAD>
        matrix(const matrix<ATAD>& m) {
            this->getMemoryforVal(m.rows(), m.cols());

            #pragma omp parallel for collapse(2) if(m.rows() >= 64 || m.cols() >= 64)
            for (int i = 0; i < this->row; ++i) {
                for (int j = 0; j < this->col; ++j) {
                    if constexpr(std::is_class_v<ATAD>) {
                        if constexpr (std::is_same_v<ATAD, std::complex<typename ATAD::value_type>>) {
                            *(val + this->col * i + j) = static_cast<DATA>(std::real(m(i, j)));
                        } 
                    }   else if constexpr (!std::is_same_v<DATA, ATAD>) {
                        *(val + this->col * i + j) = static_cast<DATA>(m(i, j));
                    }
                }
            }
        }

        // Move constructor
        matrix(matrix&& other) noexcept {
            // Transfer ownership of resources from source to destination
            this->val = std::move(other.val);
            this->row = other.row;
            this->col = other.col;
            this->first = other.first;
            this->last = other.last;

            // Reset source to a valid state
            other.val = NULL;
            other.row = 0;
            other.col = 0;
            other.first = NULL;
            other.last = NULL;
        }
        // Move constructor for handling different type matrices
        template<typename ATAD>
        matrix(matrix<ATAD>&& other) noexcept {
            this->getMemoryforVal(other.rows(), other.cols());

            #pragma omp parallel for collapse(2) if(other.rows() >= 100 || other.cols() >= 100)
            for (int i = 0; i < this->row; ++i) {
                for (int j = 0; j < this->col; ++j) {
                    if constexpr(std::is_class_v<ATAD>) {
                        if constexpr (std::is_same_v<ATAD, std::complex<typename ATAD::value_type>>) {
                            *(val + this->col * i + j) = static_cast<DATA>(std::real(std::move(other(i, j))));
                        } else if constexpr(std::is_same_v<ATAD,std::complex<DATA>>) {
                            *(val + this->col * i + j) = std::real(std::move(other(i, j)));
                        }
                    } else if constexpr (!std::is_same_v<DATA, ATAD>) {
                        *(val + this->col * i + j) = static_cast<DATA>(std::move(other(i, j)));
                    }
                }
            }
            // Reset source to a valid state
            other.delMemoryforVal();
        }

        //initializer list
        matrix(std::initializer_list<std::initializer_list<DATA>> list) {
            if(list.size() == 0 || list.begin()->size() == 0)
                matrix();
            else {
                this->getMemoryforVal(list.size(), list.begin()->size());
                
                int i=0, j=0;
                for(const auto& ROW : list) {
                    for(const auto& elem : ROW) {
                        *(val + (this->col)*i + j) = static_cast<DATA>(elem);
                        ++j;
                    }
                    j=0;
                    ++i;
                }
            }
        }

        // Operator overloading for operator<< for chaining values into the matrix
        matrix<DATA>& operator<<(const DATA& value);
        template<typename ATAD, typename std::enable_if_t<is_numeric_v<ATAD>>>
        matrix<DATA>& operator<<(const ATAD& value);

        // insert/update all the elements in row major form into the internal data structure
        void insertAll(int r=-1, int c=-1);

        // insert value at rth row and cth column
        void insertAt(DATA, int, int);

        // update using flattened array
        void updateWithArray(DATA*, int, int);

        // display contents in a 2d grid form
        void display(const std::string msg="Matrix:-") const;
        
        //set `subMatrix` values
        void setSubMatrix(int,int,int,int, const matrix&);
        void setSubMatrix(range,range, const matrix&);

        void iota(int start=int());

        ~matrix() noexcept {
            this->delMemoryforVal();
        }

        /////// MATRIX OPERATIONS
        matrix<DATA> &operator+=(matrix const& );
        matrix<DATA> &operator+=(const DATA);
        matrix<DATA> &operator-=(matrix const& );
        matrix<DATA> &operator-=(const DATA);
        template<typename ATAD>
        matrix<DATA> &operator*=(const ATAD);
        template<typename ATAD>
        matrix<DATA> &operator/=(const ATAD);
    
        // Index operator
        inline  DATA& operator()(const int, const int);
        inline const DATA& operator()(const int, const int) const;
        matrix<DATA> operator()(const matrix<bool>&);

        //Assignment operator
        matrix<DATA> &operator=(const matrix<DATA>& m1) {
            this->changeDims(m1.rows(), m1.cols());
            this->updateWithArray(m1.val, m1.rows(), m1.cols());
            return *this;
        }
        template<typename ATAD>
        matrix<DATA> &operator=(const matrix<ATAD>& m1) {
            this->changeDims(m1.rows(), m1.cols());
             if constexpr ( std::is_same_v<ATAD,std::complex<DATA>>) {
                #pragma omp parallel for collapse(2) if(m1.rows() >= 100 || m1.cols() >= 100)
                for(int i=0; i<m1.rows(); ++i)
                    for(int j=0; j<m1.cols(); ++j) 
                        *(val + i*(this->cols()) + j) = std::real(m1(i,j));
            } else if constexpr(!std::is_same_v<DATA, ATAD>) {
                #pragma omp parallel for collapse(2) if(m1.rows() >= 100 || m1.cols() >= 100)
                for(int i=0; i<m1.rows(); ++i)
                    for(int j=0; j<m1.cols(); ++j) 
                        *(val + i*(this->cols()) + j) = static_cast<DATA>(m1(i,j));
            }
            return *this;
        }   
    
        // Reshape
        matrix<DATA> reshape(int newRow, int newCol);

        // Transpose operation
        matrix<DATA> operator~() const;
        matrix<DATA> transpose();
        matrix<DATA> T(){ return this->transpose();}
        // Adjoint operation
        matrix<DATA> adjoint();

        /// Slice operation
        matrix<DATA> slice(int, int, int, int);
        matrix<DATA> operator()(range, range);

        // Element-wise exponent operation
        matrix<DATA> operator^(int);
        template<typename ATAD, typename std::enable_if_t<std::is_arithmetic_v<ATAD>>>
        matrix<DATA> operator^(ATAD);
        
        /// Swap Operations
        void swapRows(int,int);
        void swapCols(int, int);
        
        /// Generate Augmented matrix (vertical stack or horizontal stack)
        matrix<DATA> hStack(matrix const& ); // horizontal stack - hStack
        matrix<DATA> vStack(matrix const& ); // vertical  stack - vStack
        matrix<DATA> stack(matrix const& obj, bool vert=false); //generalized stack - stack

        /// aggregate functions (more to come)
        matrix<DATA> max(int dim=-1);
        matrix<DATA> argmax(int dim=-1);
        matrix<DATA> min(int dim=-1);
        matrix<DATA> argmin(int dim=-1);

        /// matrix inverse operation
        matrix<DATA> inv(); //experimental

        // solve Ax = b
        matrix<DATA> solve(const matrix<DATA>&); //experimental

        // get determinant //gaussian elimination
        double det(bool fullPivot=false);
        double determinant(bool fullPivot=false) {
            return this->det(fullPivot);
        }

        /// QUERY methods
        bool isSquare() const { if(this->col == this->row) return true; else return false;}
        bool isSymmetric() const;
        DATA item() const;
        bool isComparable(const matrix<DATA>&) const;
        bool isMatMulDefined(const matrix<DATA>&) const;
        bool all(bool value) const;
        bool isany(bool value) const;
        std::vector<std::vector<DATA>> toVector();

        /// FILE OPERATIONS I/O
        bool saveMatrix(const std::string&, const std::string& folderpath=SAVEPATH);
        bool loadMatrix(const std::string&, const std::string& folderpath=SAVEPATH);

        
};


//// NON-MEMBER OPERATIONS DECLARATIONS ///
matrix<bool> operator!(const matrix<bool>&);
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

template<typename DATA, typename ATAD>
matrix<DATA> operator*(const matrix<DATA>&, const ATAD);
template<typename DATA, typename ATAD>
matrix<DATA> operator*(const ATAD, const matrix<DATA>&);
template<typename DATA>
matrix<DATA> operator*(const matrix<DATA>&, const matrix<DATA>&);

template<typename DATA>
matrix<DATA> operator&(const matrix<DATA>&, const matrix<DATA>&);

template<typename DATA>
matrix<bool> operator==(const matrix<DATA>&, const matrix<DATA>&);
template<typename DATA, typename ATAD,\
         typename  std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator==(const matrix<DATA>&, const ATAD);
template<typename DATA, typename ATAD,\
        typename std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator==(const ATAD,const matrix<DATA>&);

template<typename DATA>
matrix<bool> operator<(const matrix<DATA>&, const matrix<DATA>&);
template<typename DATA, typename ATAD,\
        typename std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator<(const matrix<DATA>&, const ATAD);
template<typename DATA, typename ATAD,\
        typename std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator<(const ATAD,const matrix<DATA>&);

template<typename DATA>
matrix<bool> operator>(const matrix<DATA>&, const matrix<DATA>&);
template<typename DATA, typename ATAD,\
        typename std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator>(const matrix<DATA>&, const ATAD);
template<typename DATA, typename ATAD,\
        typename std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator>(const ATAD,const matrix<DATA>&);

template<typename DATA>
matrix<bool> operator<=(const matrix<DATA>&, const matrix<DATA>&);
template<typename DATA, typename ATAD,\
        typename std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator<=(const matrix<DATA>&, const ATAD);
template<typename DATA, typename ATAD,\
        typename std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator<=(const ATAD,const matrix<DATA>&);

template<typename DATA>
matrix<bool> operator>=(const matrix<DATA>&, const matrix<DATA>&);
template<typename DATA, typename ATAD,\
        typename std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator>=(const matrix<DATA>&, const ATAD);
template<typename DATA, typename ATAD,\
        typename std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator>=(const ATAD,const matrix<DATA>&);

template<typename DATA, typename ATAD>
matrix<DATA> operator/(const matrix<DATA>&, const ATAD);

template<typename DATA>
matrix<DATA> eye(int);

template<typename DATA>
matrix<DATA> diagonal(int, DATA);
template<typename DATA>
matrix<DATA> diag(const matrix<DATA>&, int shift=0);

matrix<double> upper_triangle_matrix(int, double mean=0., double std=1.);
matrix<double> lower_triangle_matrix(int, double mean=0., double std=1.);
matrix<double> utm(int, double mean=0., double std=1.);
matrix<double> ltm(int, double mean=0., double std=1.);
matrix<double> triu(int, double mean=0., double std=1.);
matrix<double> tril(int, double mean=0., double std=1.);

template<typename DATA>
bool is_triangular(matrix<DATA>&);

matrix<double> zeros(int);
matrix<double> zeros(int,int);
template<typename DATA>
matrix<DATA> zeros_like(const matrix<DATA>&);
template<typename DATA>
matrix<DATA> matrix_like(const matrix<DATA>&);

template<typename DATA>
matrix<DATA> ones(int);
template<typename DATA>
matrix<DATA> ones(int,int);
matrix<double> ones(int);
matrix<double> ones(int,int);

matrix<double> randomUniform(int, double minVal=0., double maxVal=1.);
matrix<double> randomUniform(int, int, double minVal=0., double maxVal=1.);

matrix<int> randomUniformInt(int, int, int);
matrix<int> randomUniformInt(int, int, int, int);

template<typename DATA,typename std::enable_if_t<std::is_floating_point_v<DATA>>>
matrix<DATA> randomNormal(int, DATA mean=0., DATA std=1.);
template<typename DATA,typename std::enable_if_t<std::is_floating_point_v<DATA>>>
matrix<DATA> randomNormal(int, int, DATA mean=0., DATA std=1.);

matrix<double> randomNormal(int, int, double mean=0., double std=1.);
matrix<double> randomNormal(int, double mean=0., double std=1.);

/// Non-member operations declarations end ///


/// Get 2D Vector
template<typename DATA>
std::vector<std::vector<DATA>> matrix<DATA>::toVector() {
    std::vector<std::vector<DATA>> result(this->rows(), std::vector<DATA>(this->cols()));
    int Cols = this->cols();

    #pragma omp parallel for if(this->rows() >= 100 || this->cols() >= 100)
    for(DATA *itr = this->begin(); itr != this->end(); ++itr)
        {
            int rowIdx = (itr - this->begin())/Cols;
            int colIdx = (itr - this->begin())%Cols;
            result[rowIdx][colIdx] = *itr;
        }

    return result;
}

/// IOTA
template<typename DATA>
void matrix<DATA>::iota(int start) {
    if (!std::is_arithmetic<DATA>::value || std::is_same<DATA,bool>::value)
        throw std::domain_error("matrix should only be arithmetic type.");

    if(this->rows() < 1 || this->cols() < 1)
        throw std::out_of_range("matrix is an empty matrix. Inflate it with memory first.");
    
    #pragma omp parallel for if(this->rows() >= 100 || this->cols() >= 100)
    for(DATA* itr = this->first; itr != this->last; ++itr)
        *itr = start++;
}

/// RESHAPE METHOD DEFINITION
template<typename DATA>
matrix<DATA> matrix<DATA>::reshape(int newRow, int newCol) {
    if(newRow * newCol != (this->cols())*(this->rows())){
        throw std::invalid_argument("The product of dimensions do not match the product of dimensions of the given matrix.");
    }

    matrix<DATA> reshapedMatrix(newRow, newCol);
    #pragma omp parallel for if(this->cols() >= 100 || this->rows() >= 100)
    for(int i=0; i< ((this->cols()) * (this->rows())); ++i ) {
        reshapedMatrix(i/newCol, i%newCol) = val[i];
    }
    return reshapedMatrix;
}

/// Picking pivot using FULL PIVOTING 
template<typename DATA>
void matrix<DATA>::pickPivotFullPivoting(int startRow, int& pivotRow, int& pivotCol) {
    pivotRow = startRow;
    pivotCol = startRow;
    for(int i=startRow; i<this->rows(); ++i) {
        for(int j=startRow; j<this->cols(); ++j) {
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
    #pragma omp parallel for if(this->cols() >= 100)
    for(int j=0; j<col; ++j) {
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

    #pragma omp parallel for if(this->rows() >= 100)
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
    
     for(int i=0; i<minDim; ++i) {  //test : minDim replaced n here
        //finding the pivot
        int pivotRow = i;
        for(int k=i+1; k<n; ++k) {
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
        for(int j=0; j< m; ++j) {
            augMat(i, j) /=  pivot;
        }

        for(int k=0; k<n; ++k) {
            if( k != i) {
                DATA factor = augMat(k, i);

                // row combine for augMat
                for(int j=0; j<m; ++j) {
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
            for(int i=1; i<this->row*this->col; ++i)
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
                for(int j=0; j<this->col; ++j) {
                    minElem = *(val +  j);

                    for(int i=1; i<this->row; ++i)
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
               for(int i=0; i<this->row; ++i) {
                    minElem = *(val + i*(this->col));

                    for(int j=1; j<this->col; ++j) {
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
            for(int i=0; i<this->row; ++i) 
                for(int j=0; j<this->col; ++j)
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
                for(int j=0; j<this->col; ++j) {
                    int minIdx_i = 0;
                    DATA minElem = *(val + minIdx_i*(this->col) + j);
                    for(int i=1; i<this->row; ++i)
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
               for(int i=0; i<this->row; ++i) {
                    int minIdx_j = 0;
                    DATA minElem = *(val + i*(this->col) + minIdx_j);
                    
                    for(int j=1; j<this->col; ++j) {
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
            for(int i=1; i<this->row*this->col; ++i)
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

                for(int j=0; j<this->col; ++j) {
                    maxElem = *(val +  j);

                    for(int i=1; i<this->row; ++i)
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
               for(int i=0; i<this->row; ++i) {
                    maxElem = *(val + i*(this->col));

                    for(int j=1; j<this->col; ++j) {
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
            for(int i=0; i<this->row; ++i) 
                for(int j=0; j<this->col; ++j)
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

                for(int j=0; j<this->col; ++j) {
                    int maxIdx_i = 0;
                    DATA maxElem = *(val + maxIdx_i*(this->col) + j);
                    for(int i=1; i<this->row; ++i)
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
               for(int i=0; i<this->row; ++i) {
                    int maxIdx_j = 0;
                    DATA maxElem = *(val + i*(this->col) + maxIdx_j);
                    
                    for(int j=1; j<this->col; ++j) {
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

/// COMPARE DIMENSIONS
template<typename DATA>
bool matrix<DATA>::isComparable(const matrix<DATA>& m) const {
    if(this->rows() == m.rows() && this->cols() == m.cols()) {
        return true;
    }
    return false;
}
template<typename DATA>
bool matrix<DATA>::isMatMulDefined(const matrix<DATA>& m) const {
    if(this->cols() == m.rows())
        return true;
    return false;
}

template<typename DATA>
bool matrix<DATA>::isSymmetric() const{
    if(this->row == this->col)
     {
        matrix<DATA> Transpose = ~(*this);
        if((Transpose == *this).all(true))
            return true;
     }
     return false;
}

template<typename DATA>
DATA matrix<DATA>::item() const{
    if(this->row == 1  && this->col == 1) {
        return *val; 
    } else {
        throw std::invalid_argument("linear::matrix::item() - To throw an item out it is supposed to be 1x1 matrix.");
    }
}

template<typename DATA>
 bool matrix<DATA>::all(bool value) const {
    static_assert(std::is_same_v<DATA,bool>, "all() is only supported for boolean matrices.");
    
    for(int i=0; i<this->rows(); ++i) {
        for(int j=0; j<this->cols(); ++j) {
            if(value != *(val + i*(this->cols()) + j)) {
                return false;
            }
        }
    }
    return true;
}

template<typename DATA>
bool matrix<DATA>::isany(bool value) const {
    static_assert(std::is_same_v<DATA,bool>, "isany() is only supported for boolean matrices.");
    
    for(int i=0; i<this->rows(); ++i) {
        for(int j=0; j<this->cols(); ++j) {
            if(value == *(val + i*(this->cols()) + j)) {
                return true;
            }
        }
    }
    return false;
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
    for(int i=0; i<m.row; ++i) {
        for(int j=0; j<this->col; ++j)
            *(m.val + i*m.col + j) = *(val + (this->col)*i + j);

        for(int j=0; j<obj.col; ++j)
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
    for(int j=0; j<m.col; ++j) {
        for(int i=0; i<this->row; ++i)
            *(m.val + i*m.col + j) = *(val + i*m.col + j);

        for(int i=0; i<obj.row; ++i)
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
    int n = rows(); //rows
    int m = cols(); //cols
    
    matrix<DATA> this_copy(this->val, n, m); //copy of this
    double detValue = 1.0;
    double sign = 1;

    if(fullPivot) {
        for(int i=0; i<n-1; ++i) {
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

            for(int k=i+1; k<n; ++k) {
                double factor = this_copy(k,i) / pivot;
                
                for(int j=i; j<n; ++j) {
                    this_copy(k, j) -= factor * this_copy(i, j);
                }
            }
            detValue *= pivot; //multiply detValue by pivot value 
    }
    detValue *= this_copy(n-1, n-1);
    detValue *= sign;     //finally implement the sign into it
    } else { //partial pivoting

        for(int i=0; i< n-1; ++i) {
            int pivotIdx=i;
            double maxVal = abs(this_copy(i,i));
            for(int k=i+1; k<n; ++k) {
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
            
            for(int k=i+1; k<n; ++k) {
                double factor = this_copy(k,i) / pivot;
                for(int j=i; j<n; ++j)
                    this_copy(k,j) -= factor * this_copy(i,j);
            }
            detValue *= pivot;
        }
        detValue *= this_copy(n-1,n-1);
        detValue *= sign;
    } //partial pivoting ends here
    return detValue;
}

// Operator overloading for operator<< for chaining values into the matrix
template<typename DATA>
matrix<DATA>& matrix<DATA>::operator<<(const DATA& value) {
    if(cur_row >= row || cur_col >= col)
        throw std::out_of_range("operator<< - Index out of range. Matrix is filled.");
    
    this->val[cur_row*col + cur_col++] = value;
    if(cur_col == col) {
        cur_col = 0;
        ++cur_row;
    }
    return *this;
}

template<typename DATA>
template<typename ATAD, typename std::enable_if_t<is_numeric_v<ATAD>>>
matrix<DATA>& matrix<DATA>::operator<<(const ATAD& value) {
    if(cur_row >= row || cur_col >= col)
        throw std::out_of_range("operator<< - Index out of range. Matrix is filled.");
    
    if constexpr(std::is_same_v<std::complex<DATA>, std::decay_t<ATAD>>) {
        this->val[cur_row*col + cur_col++] = std::real(value);
    } else {
        this->val[cur_row*col + cur_col++] = static_cast<DATA>(value);
    }
    if(cur_col == col) {
        cur_col = 0;
        ++cur_row;
    }
    return *this;
}

/// ELEMENT-WISE EXPONENT OPERATION
template<typename DATA>
matrix<DATA> matrix<DATA>::operator^(int power) {
    matrix<DATA> m(this->row, this->col);
    #pragma omp parallel for collapse(2) if(this->rows()>=100 || this->cols() >= 100)
    for(int i=0; i<m.rows(); ++i) {
        for(int j=0; j<m.cols(); ++j) {
            DATA prod=1.;
            prod = std::pow(*(val + i*(this->col) + j), power);
            m(i,j) = prod;
        }
    }
    return m;
}
template<typename DATA>
template<typename ATAD, typename std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<DATA> matrix<DATA>::operator^(ATAD power) {
    matrix<DATA> m(this->row, this->col);

    #pragma omp parallel for collapse(2) if(this->rows()>=100 || this->cols() >=100)
    for(int i=0; i<m.rows(); ++i) {
        for(int j=0; j<m.cols(); ++j) {
            m(i,j) = std::pow(val[i*col +j],power);
        }
    }
    return m;
}

/// INDEX OPERATION
template<typename DATA>
inline  DATA& matrix<DATA>::operator()(const int r, const int c)  {
    if(r >=0 && r < this->rows()  && c >= 0 && c < this->cols()) {
        return *(val + r*this->col + c);
    } else {
        throw std::out_of_range("linear::matrix::operator() - matrix indices out of range.");
    }
}
template<typename DATA>
inline const DATA& matrix<DATA>::operator()(const int r, const int c) const {
    if(r >=0 && r < this->rows()  && c >= 0 && c < this->cols()) {
        return *(val + r*this->col + c);
    } else {
        throw std::out_of_range("linear::matrix::operator() - matrix indices out of range.");
    }
}
template<typename DATA>
matrix<DATA> matrix<DATA>::operator()(const matrix<bool>& m) {
    if(m.rows() != this->rows() || m.cols() != this->cols())
        throw std::invalid_argument("the corresponding dimensions do not match.");

    int size=0;
    for(bool* itr=m.begin(); itr != m.end(); ++itr) {
        if(*itr)
            size++;
    }

    matrix<DATA> result(1,size);
    int itr=0;
    for(int i=0; i<this->rows(); ++i){
        for(int j=0; j<this->cols(); ++j) {
            if(m(i,j))
                result(0,itr++) = (*this)(i,j);
        }
    }
    return result;
}

/// SLICE OPERATION
template<typename DATA>
matrix<DATA> matrix<DATA>::operator()(range rowRange, range colRange) {
    /*
        sister function of `slice` operation.
        It makes use of `range` type to accomplish slicing.
        EXAMPLE USAGE:-
            matrix<double> A(5,5,2.5);
            matrix<double> smolA = A(range(3), range(4)); //3x4 matrix
            matrix<double slicedA = A(range(2,3), range(2,5)); //1x3 matrix
        `end` index in range is excluded.
    */
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
            Suppose you have a 5x4  matrix of some random integer values
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

        for(int i=0; i<m.row; ++i) {
            for(int j=0; j<m.col; ++j) {
                *(m.val + i*m.col + j) = *(val + (i + x_0)*(this->col) + (j + x_1));
            }
        }
        return m;
    } else {
        throw std::invalid_argument("slicing matrix error raised - Wrong index range received. Check your index Params.");
    }
}

// setSubMatrix operation
template<typename DATA>
void matrix<DATA>::setSubMatrix(int x_0, int y_0, int x_1, int y_1, const matrix<DATA>& subMatrix) {
    int n=this->rows();
    int m=this->cols();
    bool validation1 = (this->validateParams(x_0, y_0, n)) && (this->validateParams(x_1, y_1, m));
    //bool validation2 = !((y_0 - x_0  != n) || (y_1 - x_1 != m)); 
    if (validation1) {
        int i,j;
        #pragma omp parallel for collapse(2) 
        for (i = x_0; i < y_0; ++i) {
            for (j = x_1; j < y_1; ++j) {
                (*this)(i, j) = subMatrix(i - x_0, j - x_1);
            }
        }
    } else {
        throw std::invalid_argument("linear::matrix::setSubMatrix - Wrong range indices received. Check your arguments.");
    }
}
template<typename DATA>
void matrix<DATA>::setSubMatrix(range rowRng, range colRng, const matrix<DATA>& subMatrix) {
    this->setSubMatrix(rowRng.start, rowRng.end, colRng.start, colRng.end, subMatrix);
}

/// TRANSPOSE OPERATION
template<typename DATA> //applicable for const matrices
matrix<DATA> matrix<DATA>::operator~() const {
    matrix<DATA> m(this->col, this->row);

    for(int i=0; i<m.row; ++i)
        for(int j=0; j<m.col; ++j)
            m(i,j) = *(val + j*col + i);
    return m;
}

/// TRANSPOSE OPERATION 
template<typename DATA>
matrix<DATA> matrix<DATA>::transpose() {
    matrix m = ~(*this);
    return m;
}

/// Adjoint of matrix
template<typename DATA>
matrix<DATA> matrix<DATA>::adjoint() {
    matrix<DATA> m = ~(*this);
    if constexpr(std::is_class_v<DATA>) // necessary to make this check for a class type
        if constexpr(std::is_same_v<DATA,std::complex<typename DATA::value_type>>) {
            for(DATA* itr=m.begin(); itr != m.end(); ++itr)
                (*itr).imag(-(*itr).imag());
        }
    return m;
}

/// Insertion Operations
template<typename DATA>
void matrix<DATA>::insertAll(int r, int c)  {
    if(r > -1 &&  c > -1)
        this->changeDims(r, c);
    int i,j;
    std::cout<<"\nNote: you have to insert "<<this->row*this->col<<" values. Values will be filled row-major wise in a "<<this->row<<'x'<<this->col<<" matrix.\n";
    for(i=0; i<this->row; ++i)
        for(j=0; j<this->col; ++j)
            std::cin>>*(val + (this->col)*i + j);
}
template<typename DATA>
void matrix<DATA>::insertAt(DATA value, int r, int c)  {
        if( (r>-1 && r < this->row) && (c>-1 && c<this->col)) {
            *(val + (this->col)*r + c) = value;
        } else {
            throw std::invalid_argument("linear::matrix::insertAt() - The index values exceed the dimension size of the matrix.");
        }
}
template<typename DATA>
void matrix<DATA>::updateWithArray(DATA* array, int r, int c) {
    if (r <0 || c < 0)
        throw std::invalid_argument("linear::matrix::updateWithArray() - Bad dimension values.");

    this->changeDims(r, c);
    std::copy(array, array + r*c, this->first);
}

/// Print matrix in ostream
template<typename DATA>
void matrix<DATA>::display(const std::string msg)  const{
    //experimental code
    int i,j;
    std::cout<<'\n'<<msg<<'\n';

    // zero size matrix display
    if(this->rows() == 0 || this->cols() == 0 ) {
        std::cout<<"(empty matrix)\n";
        return;
    }

    int max_precision = MATRIX_PRECISION;
    int padding = 1;
    int maxDigits = 1;
    // if it is a bool matrix use different logic
    if constexpr(std::is_same_v<DATA,bool>) {
        for(int i=0; i<rows(); i++) {
            for(int j=0; j<cols();j++)
                std::cout << std::setw(5)<< std::boolalpha << (*(val + (col) * i + j)) << " ";
            std::cout<<'\n';
        }
    } else if constexpr(std::is_integral_v<DATA>) {
            padding = 0;
            for (i = 0; i < rows(); ++i) {
                for (j = 0; j < cols(); ++j) {
                    std::stringstream stream;
                    stream << *(val + (col) * i + j);
                    std::string str = stream.str();
                    maxDigits = std::max(maxDigits, static_cast<int>(str.length()));
                }
            }
            // Set the width based on the maximum number of digits
            int width = maxDigits + padding;
            for (i = 0; i < rows(); ++i) {
                for (j = 0; j < cols(); ++j) {
                    std::cout << std::setw(width) << *(val + (col) * i + j) << " ";
                }
                std::cout << "\n";
            }
    } else {
        // Find the maximum number of digits in the matrix
        for (i = 0; i < rows(); ++i) {
            for (j = 0; j < cols(); ++j) {
                std::stringstream stream;
                stream << std::fixed << std::setprecision(max_precision) << *(val + (col) * i + j);
                std::string str = stream.str();

                size_t pos = str.find_last_not_of('0');
                if (pos != std::string::npos && str[pos] == '.')
                    pos--;

                maxDigits = std::max(maxDigits, static_cast<int>(pos + 1));
            }
        }
        // Set the width based on the maximum number of digits
        int width = maxDigits + padding;
        for (i = 0; i < rows(); ++i) {
            for (j = 0; j < cols(); ++j) {
                std::stringstream stream;
                stream << std::fixed << std::setprecision(max_precision) << *(val + (col) * i + j);
                std::string str = stream.str();

                size_t pos = str.find_last_not_of('0');
                if (pos != std::string::npos && str[pos] == '.')
                    pos--;
                std::cout << std::setw(width) << str.substr(0, pos + 1);
            }
            std::cout << "\n";
        }
    } //if not bool type else condition
}

/// File operation on saving a matrix
template<typename DATA>
bool matrix<DATA>::saveMatrix(const std::string& filename, const std::string& folderpath) {
    std::string fullpath = folderpath + '/' + filename + F_EXT;
    std::ofstream saveFile(fullpath);

    if(saveFile.is_open()) {
        // saving the dimensions
        saveFile<<row<<" "<<col<<"\n";

        // saving the flattened array elements (row major)
        for(int i=0; i<row; ++i) {
            for(int j=0; j<col; ++j) {
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
bool matrix<DATA>::loadMatrix(const std::string& filename, const std::string& folderpath) {
    std::string fullpath = folderpath + '/' + filename + F_EXT;
    std::ifstream loadFile(fullpath);

    if(loadFile.is_open()) {
        int loadR, loadC;
        loadFile>>loadR>>loadC;

        //resize the matrix
        this->changeDims(loadR, loadC);

        //reading matrix values from file
        for(int i=0; i<this->row; ++i){
            for(int j=0; j<this->col; ++j) {
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
    matrix<DATA> m(n,n,0);
    
    for(int i=0; i<n; ++i)
        for(int j=0; j<n; ++j)
            {
                if(i==j)
                    m.insertAt(value, i, j);
            }
    return m;
}
//diagonam matrix generator : using a vector matrix
template<typename DATA>
matrix<DATA> diag(const matrix<DATA> &m1, int shift) {
    int R = m1.rows();
    int C = m1.cols();

    if(R<0 || C<0)
        throw std::domain_error("diag() - Input matrix is empty.");
    if(!((C == 1) ^ (R == 1)))
        throw std::invalid_argument("diagonal() - Input matrix is not a vector.");
    
    int abs_shift = ((shift<0)?-shift:shift);
    int SIZE = ((C==1)?R:C) + abs_shift;
    matrix<DATA> m(SIZE, SIZE, (DATA)0.);
    for(int i=0; i<((C==1)?R:C); ++i) {
        if(shift<0) {
            m(i+abs_shift,i) = ((C==1)?m1(i,0):m1(0,i));
        } else {
            m(i,i+abs_shift) = ((C==1)?m1(i,0):m1(0,i));
        }
    }
    return m;
}

// Identity matrix of size n
template<typename DATA>
matrix<DATA> eye(int n) {
    matrix<DATA> m(n);
    #pragma omp parallel for collapse(2) if(n>=100)
    for(int i=0; i<n; ++i)
        for(int j=0; j<n; ++j) {
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
    int n = M.rows();
    int m = M.cols();

    // machine epsilon
    DATA epsilon = std::numeric_limits<DATA>::epsilon();
    bool upper=true, lower=true;

    //check upper triangular
    for(int i=1; i<n; ++i) {
        for(int j=0; j<i && j<m; ++j) {
            if(std::abs(M(i,j)) > epsilon) {
                upper = false;
                break;
            }
        }
    } // upper triangular

    //check lower triangular
    for(int i=0; i<n; ++i) {
        for(int j=i+1; j<m; ++j) {
            if( std::abs(M(i,j)) > epsilon ) {
                lower = false;
                break;
            }
        }
    } // lower triangular
    return (upper || lower);
}

// all data values 
template<typename DATA>
matrix<DATA> zeros(int n) {
    matrix<DATA> _0s(n,n, DATA(0));
    return _0s;
}

// all data values
template<typename DATA>
matrix<DATA> zeros(int n, int m) {
    matrix<DATA> _0s(n,m, DATA(0));
    return _0s;
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
template<typename DATA>
matrix<DATA> zeros_like(const matrix<DATA>& m) {
    matrix<DATA> _0s(m.rows(), m.cols(), DATA(0.));
    return _0s;
}
template<typename DATA>
matrix<DATA> matrix_like(const matrix<DATA>& m) {
    matrix<DATA> _0s(m.rows(), m.cols());
    return _0s;
}
template<typename DATA>
matrix<DATA> ones_like(const matrix<DATA>& m) {
    matrix<DATA> _1s(m.rows(), m.cols(), DATA(1.));
    return _1s;
}

//all data type values for ones
template<typename DATA>
matrix<DATA> ones(int n) {
    matrix<DATA> _1s(n,n,(DATA)1);
    return _1s;
}

template<typename DATA>
matrix<DATA> ones(int n, int m) {
    matrix<DATA> _1s(n,m,(DATA)1);
    return _1s;
}

// double ones matrices
matrix<double> ones(int n) {
    matrix<double> _1s(n,n,1.);
    return _1s;
}

// double rectangle matrices of ones
matrix<double> ones(int n, int m) {
    matrix<double> _1s(n,m,1.);
    return _1s;
}


// random square matrix
matrix<double> randomUniform(int n, double minVal, double maxVal) {
    matrix<double> mat(n);

    std::random_device dev;
    std::mt19937 generator(dev());
    std::uniform_real_distribution<double> distribution(minVal, maxVal);

    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j)
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

    for(int i=0; i<n; ++i){
        for(int j=0; j<m; ++j)
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

    for(int i=0; i<n; ++i) {
        for(int j=0; j<n; ++j)
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

    for(int i=0; i<n; ++i) {
        for(int j=0; j<m; ++j)
            mat(i,j) = distribution(generator);
    }
    return mat;
}

//random normal double specialisation
matrix<double> randomNormal(int n, double mean, double std) {
    matrix<double> mat(n);

    std::random_device dev;
    std::mt19937 generator(dev());
    std::normal_distribution<double> distribution(mean, std);

    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j)
            mat(i,j) = distribution(generator);
    }
    return mat;
}

matrix<double> randomNormal(int n, int m, double mean, double std) {
    matrix<double> mat(n,m);

    std::random_device dev;
    std::mt19937 generator(dev());
    std::normal_distribution<double> distribution(mean, std);

    for(int i=0; i<n; ++i){
        for(int j=0; j<m; ++j)
            mat(i,j) = distribution(generator);
    }
    return mat;
}

// random nxm matrix from normal distribution
template<typename DATA,typename std::enable_if_t<std::is_floating_point_v<DATA>>>
matrix<DATA> randomNormal(int n, int m, DATA mean, DATA std) {
    matrix<DATA> mat(n,m);

    std::random_device dev;
    std::mt19937 generator(dev());
    std::normal_distribution<DATA> distribution(mean, std);

    for(int i=0; i<n; ++i){
        for(int j=0; j<m; ++j)
            mat(i,j) = distribution(generator);
    }
    return mat;
}

// random nxn matrix from normal distribution
template<typename DATA,typename std::enable_if_t<std::is_floating_point_v<DATA>>>
matrix<DATA> randomNormal(int n, DATA mean, DATA std) {
    matrix<DATA> mat(n);

    std::random_device dev;
    std::mt19937 generator(dev());
    std::normal_distribution<DATA> distribution(mean, std);

    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j)
            mat(i,j) = distribution(generator);
    }
    return mat;
}

// not operator on a boolean matrix
matrix<bool> operator!(const matrix<bool> &m) {
    matrix<bool> result(m.rows(), m.cols());
    auto itRez = result.begin();
    for(auto it = m.begin(); it != m.end(); ++it, ++itRez) {
        *itRez = !(*it);
    }
    return result;
}

template<typename DATA>
template<typename ATAD>
matrix<DATA> &matrix<DATA>::operator*=(const ATAD value) {
    DATA newVal;
    if constexpr(std::is_same_v<ATAD, std::complex<DATA>>) {
        newVal = static_cast<DATA>(std::real(value));
    } else {
        newVal = static_cast<DATA>(value);
    }

    #pragma omp parallel for if(row*col >= 100)
    for(int i=0; i < row*col; ++i)
        *(val + i) *= newVal;
    return *this;
}
template<typename DATA>
template<typename ATAD>
matrix<DATA> &matrix<DATA>::operator/=(const ATAD value) {
    DATA newVal;
    if constexpr(std::is_same_v<ATAD, std::complex<DATA>>) {
        newVal = static_cast<DATA>(std::real(value));
    } else {
        newVal = static_cast<DATA>(value);
    }
    #pragma omp parallel for if(row*col >= 100)
    for(int i=0; i < row*col; ++i)
        *(val + i) /= newVal;
    return *this;
}

template<typename DATA>
matrix<DATA> &matrix<DATA>::operator-=(const matrix<DATA>& m1) {
    if(!this->isComparable(m1))
        throw std::invalid_argument("Dimensions do not match.");
    else {
        #pragma omp parallel for if(row*col >= 100)
        for(int i=0; i<this->rows()*this->cols(); ++i)
            *(val + i) -= m1(i/m1.cols(), i%m1.cols());
    }
    return *this;
}
template<typename DATA>
matrix<DATA> &matrix<DATA>::operator-=(const DATA value) {

    #pragma omp parallel for if(row*col >= 100)
    for(int i=0; i<this->rows()*this->cols(); ++i)
        *(val + i) -= value;
    return *this;
}

template<typename DATA, typename ATAD>
matrix<DATA> operator/(const matrix<DATA>& m1, const ATAD value) {
    matrix<DATA> result = m1;
    if constexpr(std::is_same_v<ATAD, std::complex<DATA>>) {
        result /= static_cast<DATA>(std::real(value));
    } else {
        result /= static_cast<DATA>(value);
    }
    return result;
}

template<typename DATA, typename ATAD>
matrix<DATA> operator*(const matrix<DATA>& m1, const ATAD value) {
    matrix<DATA> result = m1;
    if constexpr(std::is_same_v<ATAD, std::complex<DATA>>) {
        result *= static_cast<DATA>(std::real(value));
    } else {
        result *= static_cast<DATA>(value);
    }
    return result;
}
template<typename DATA, typename ATAD>
matrix<DATA> operator*(const ATAD val, const matrix<DATA>& m1) {
    return m1 * val;
}

template<typename DATA>
matrix<DATA> operator*(const matrix<DATA>& m1, const matrix<DATA>& m2) {
    // element-wise multiplication 
    // NOT MATRIX MULTIPLICATION
    if(!(m1.isComparable(m2))) {
        throw std::invalid_argument("Corresponding dimensions do not match for element-wise multiplication.");
    }
    matrix<DATA> product(m1.rows(), m1.cols(), (DATA)1);
    #pragma omp parallel for if(m1.rows()*m1.cols() >= 100)
    for(int i=0; i<m1.rows()*m1.cols(); ++i){
        product(i/m1.cols(), i%m1.cols()) *= (m1(i/m1.cols(), i%m1.cols()) * m2(i/m2.cols(), i%m2.cols()));
    }
    return product;
}

template<typename DATA>
matrix<bool> operator==(const matrix<DATA>& m1, const matrix<DATA>& m2) {
    if(!m1.isComparable(m2))
        throw std::domain_error("matrix - corresponding dimensions must match");
    matrix<bool> res(m1.rows(), m1.cols(), true);

    #pragma omp parallel for collapse(2) if(m1.rows() >= 100 || m1.cols() >= 100)
    for(int i=0; i<m1.rows(); ++i)
        for(int j=0; j<m1.cols(); ++j)
                if(std::abs(m1(i,j) - m2(i,j)) > PRECISION_TOL(DATA) )
                res(i,j) = false;    
    return res;
}
template<typename DATA, typename ATAD,\
typename = std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator==(const matrix<DATA>& m1, const ATAD value) {
    matrix<bool> res(m1.rows(), m1.cols(), true);
    DATA tval;

    if constexpr(!std::is_same_v<DATA,ATAD>) {
            tval = static_cast<DATA>(value);
    } else {
        tval = value;
    }
    #pragma omp parallel for if(m1.rows() >= 100 || m1.cols() >= 100)
    for(int i=0; i<m1.rows(); i++)
        for(int j=0; j<m1.cols(); j++) {
            if(std::abs(m1(i,j) - tval) > PRECISION_TOL(DATA))
                res(i,j) = false;
        }    
    return res;
}
template<typename DATA, typename ATAD,\
typename = std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator==(const ATAD value, const matrix<DATA>& m1) {
    return (m1==value);
}

template<typename DATA>
matrix<bool> operator<(const matrix<DATA>& m1, const matrix<DATA>& m2) {
    if(!m1.isComparable(m2))
        throw std::domain_error("matrix - corresponding dimensions must match");
    matrix<bool> res(m1.rows(), m1.cols());

    #pragma omp parallel for if(m1.rows() >= 100 || m1.cols() >= 100)
    for(int i=0; i<m1.rows(); i++)
        for(int j=0; j<m1.cols(); j++)
                res(i,j) = m1(i,j)<m2(i,j);
    return res;
}
template<typename DATA, typename ATAD,\
typename = std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator<(const matrix<DATA>& m1, const ATAD value) {
    matrix<bool> res(m1.rows(), m1.cols());
    DATA tval;

    if constexpr(!std::is_same_v<DATA,ATAD>) {
            tval = static_cast<DATA>(value);
    } else {
        tval = value;
    }
    #pragma omp parallel for if(m1.rows() >= 100 || m1.cols() >= 100) 
    for(int i=0; i<m1.rows(); i++)
        for(int j=0; j<m1.cols(); j++) {
                res(i,j) = m1(i,j) < tval;
        }    
    return res;
}
template<typename DATA, typename ATAD,\
typename = std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator<(const ATAD value, const matrix<DATA>& m1) {
    return (m1>value);
}

template<typename DATA>
matrix<bool> operator>(const matrix<DATA>& m1, const matrix<DATA>& m2) {
    if(!m1.isComparable(m2))
        throw std::domain_error("matrix - corresponding dimensions must match");
    matrix<bool> res(m1.rows(), m1.cols());

    #pragma omp parallel for if(m1.rows() >= 100 || m1.cols() >= 100)
    for(int i=0; i<m1.rows(); i++)
        for(int j=0; j<m1.cols(); j++)
                res(i,j) = m1(i,j)>m2(i,j);
    return res;
}
template<typename DATA, typename ATAD,\
typename = std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator>(const matrix<DATA>& m1, const ATAD value) {
    matrix<bool> res(m1.rows(), m1.cols());
    DATA tval;

    if constexpr(!std::is_same_v<DATA,ATAD>) {
            tval = static_cast<DATA>(value);
    } else {
        tval = value;
    }
    #pragma omp parallel for if(m1.rows() >= 100 || m1.cols() >= 100) 
    for(int i=0; i<m1.rows(); i++)
        for(int j=0; j<m1.cols(); j++) {
                res(i,j) = m1(i,j) > tval;
        }    
    return res;
}
template<typename DATA, typename ATAD,\
typename = std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator>(const ATAD value, const matrix<DATA>& m1) {
    matrix<bool> res(m1.rows(), m1.cols());
    DATA tval;

    if constexpr(!std::is_same_v<DATA,ATAD>) {
            tval = static_cast<DATA>(value);
    } else {
        tval = value;
    }
    #pragma omp parallel for if(m1.rows() >= 100 || m1.cols() >= 100)
    for(int i=0; i<m1.rows(); i++)
        for(int j=0; j<m1.cols(); j++) {
                res(i,j) = m1(i,j) < tval;
        }    
    return res;
}

template<typename DATA>
matrix<bool> operator>=(const matrix<DATA>& m1, const matrix<DATA>& m2) {
    if(!m1.isComparable(m2))
        throw std::domain_error("matrix - corresponding dimensions must match");
    matrix<bool> res(m1.rows(), m1.cols());

    #pragma omp parallel for if(m1.rows() >= 100 || m1.cols() >= 100)
    for(int i=0; i<m1.rows(); i++)
        for(int j=0; j<m1.cols(); j++)
                res(i,j) = m1(i,j)>=m2(i,j);
    return res;
}
template<typename DATA, typename ATAD,\
typename = std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator>=(const matrix<DATA>& m1, const ATAD value) {
    matrix<bool> res(m1.rows(), m1.cols());
    DATA tval;

    if constexpr(!std::is_same_v<DATA,ATAD>) {
            tval = static_cast<DATA>(value);
    } else {
        tval = value;
    }
    #pragma omp parallel for if(m1.rows() >= 100 || m1.cols() >= 100) 
    for(int i=0; i<m1.rows(); i++)
        for(int j=0; j<m1.cols(); j++) {
                res(i,j) = m1(i,j) >= tval;
        }    
    return res;
}
template<typename DATA, typename ATAD,\
typename = std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator>=(const ATAD value, const matrix<DATA>& m1) {
    matrix<bool> res(m1.rows(), m1.cols());
    DATA tval;

    if constexpr(!std::is_same_v<DATA,ATAD>) {
            tval = static_cast<DATA>(value);
    } else {
        tval = value;
    }
    #pragma omp parallel for if(m1.rows() >= 100 || m1.cols() >= 100)
    for(int i=0; i<m1.rows(); i++)
        for(int j=0; j<m1.cols(); j++) {
                res(i,j) = m1(i,j) <= tval;
        }    
    return res;
}

template<typename DATA>
matrix<bool> operator<=(const matrix<DATA>& m1, const matrix<DATA>& m2) {
    if(!m1.isComparable(m2))
        throw std::domain_error("matrix - corresponding dimensions must match");
    matrix<bool> res(m1.rows(), m1.cols());

    #pragma omp parallel for if(m1.rows() >= 100 || m1.cols() >= 100)
    for(int i=0; i<m1.rows(); i++)
        for(int j=0; j<m1.cols(); j++)
                res(i,j) = m1(i,j)<=m2(i,j);
    return res;
}
template<typename DATA, typename ATAD,\
typename = std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator<=(const matrix<DATA>& m1, const ATAD value) {
    matrix<bool> res(m1.rows(), m1.cols());
    DATA tval;

    if constexpr(!std::is_same_v<DATA,ATAD>) {
            tval = static_cast<DATA>(value);
    } else {
        tval = value;
    }
    #pragma omp parallel for if(m1.rows() >= 100 || m1.cols() >= 100) 
    for(int i=0; i<m1.rows(); i++)
        for(int j=0; j<m1.cols(); j++) {
                res(i,j) = m1(i,j) <= tval;
        }    
    return res;
}
template<typename DATA, typename ATAD,\
typename = std::enable_if_t<std::is_arithmetic_v<ATAD>>>
matrix<bool> operator<=(const ATAD value, const matrix<DATA>& m1) {
    matrix<bool> res(m1.rows(), m1.cols());
    DATA tval;

    if constexpr(!std::is_same_v<DATA,ATAD>) {
            tval = static_cast<DATA>(value);
    } else {
        tval = value;
    }
    #pragma omp parallel for if(m1.rows() >= 100 || m1.cols() >= 100)
    for(int i=0; i<m1.rows(); i++)
        for(int j=0; j<m1.cols(); j++) {
                res(i,j) = m1(i,j) >= tval;
        }    
    return res;
}

template<typename DATA>
matrix<DATA>& matrix<DATA>::operator+=(const matrix<DATA>& m1) {
    if (!this->isComparable(m1))
        throw std::invalid_argument("Dimensions do not match.");
    else {
        int Size = this->rows() * this->cols();

        #ifdef __AVX__
        if constexpr (std::is_same_v<DATA, float>) {
            for (int i = 0; i < Size; i += 8) {
                __m256 vec1 = _mm256_loadu_ps(this->val + i);
                __m256 vec2 = _mm256_loadu_ps(&m1(i / m1.cols(), i % m1.cols()));
                vec1 = _mm256_add_ps(vec1, vec2);
                _mm256_storeu_ps(this->val + i, vec1);
            }
        } else if constexpr (std::is_same_v<DATA, double>) {
            for (int i = 0; i < Size; i += 4) {
                __m256d vec1 = _mm256_loadu_pd(this->val + i);
                __m256d vec2 = _mm256_loadu_pd(&m1(i / m1.cols(), i % m1.cols()));
                vec1 = _mm256_add_pd(vec1, vec2);
                _mm256_storeu_pd(this->val + i, vec1);
            }
        } else {
            
            for (int i = 0; i < Size; ++i)
                *(this->val + i) += m1(i / m1.cols(), i % m1.cols());
        }
        #else
       
        for (int i = 0; i < Size; ++i)
            *(this->val + i) += m1(i / m1.cols(), i % m1.cols());
        #endif
    }
    return *this;
}
template<typename DATA>
matrix<DATA>& matrix<DATA>::operator+=(const DATA value) {
    int Size = this->rows() * this->cols();

    #ifdef __AVX__
    if constexpr (std::is_same_v<DATA, float>) {
        for(int i=0; i<Size; i+=8) {
            __m256 vec = _mm256_loadu_ps(this->val + i);
            __m256 valueVec = _mm256_set1_ps(static_cast<float>(value));
            vec = _mm256_add_ps(vec, valueVec);
            _mm256_storeu_ps(this->val + i, vec);
        }
    } else if constexpr (std::is_same_v<DATA, double>) {
        for(int i=0; i<Size; i+=4) {
            __m256d vec = _mm256_loadu_pd(this->val + i);
            __m256d valueVec = _mm256_set1_pd(value);
            vec = _mm256_add_pd(vec, valueVec);
            _mm256_storeu_pd(this->val + i, vec);
        }
    } else {
        for(int i=0; i<Size; ++i) {
            *(this->val + i) += value;
        }
    }
    #else
    for(int i=0; i<Size; ++i) {
        *(this->val + i) += value;
    }
    #endif

    return *this;
}

template<typename DATA>
matrix<DATA> operator+(const matrix<DATA>& m1, const matrix<DATA>& m2) {
   if(!(m1.isComparable(m2))) {
        throw std::invalid_argument("corresponding dimensions do not match for addition.");
   }
   matrix<DATA> m = m1;
   m += m2;
   return m;
}

template<typename DATA>
matrix<DATA> operator+(const matrix<DATA>& m1, const double value) {
    int Size = m1.rows() * m1.cols();
    matrix<DATA> resMat = m1;

    #ifdef __AVX__
    if constexpr (std::is_same_v<DATA, float>) {
        for(int i=0; i<Size; i+=8) {
            __m256 vec = _mm256_loadu_ps(&resMat(i/resMat.cols(), i%resMat.cols()));
            __m256 valueVec = _mm256_set1_ps(static_cast<float>(value));
            vec = _mm256_add_ps(vec, valueVec);
            _mm256_storeu_ps(&resMat(i/resMat.cols(), i%resMat.cols()), vec);
        }
    } else if constexpr (std::is_same_v<DATA, double>) {
       for(int i=0; i<Size; i+=4) {
            __m256d vec = _mm256_loadu_pd(&resMat(i/resMat.cols(), i%resMat.cols()));
            __m256d valueVec = _mm256_set1_pd(value);
            vec = _mm256_add_pd(vec, valueVec);
            _mm256_storeu_pd(&resMat(i/resMat.cols(), i%resMat.cols()), vec);
       }
    } else {
        #pragma omp parallel for 
        for(int i=0; i<Size; ++i) {
            DATA temp = resMat(i/resMat.cols(), i%resMat.cols());
            temp += value;
            resMat(i/resMat.cols(), i%resMat.cols()) = temp;
        }
    }
    #else //if intrinsics not present
        for(int i=0; i<Size; ++i) {
            DATA temp = resMat(i/resMat.cols(), i%resMat.cols());
            temp += value;
            resMat(i/resMat.cols(), i%resMat.cols()) = temp;
        }
    #endif
    return resMat;
}
template<typename DATA>
matrix<DATA> operator+(const double value, const matrix<DATA>& m2) {
    return m2+value;
}

template<typename DATA>
matrix<DATA> operator-(const matrix<DATA>& m1, const matrix<DATA>& m2){
    if(!(m1.isComparable(m2))) {
        throw std::invalid_argument("corresponding dimensions do not match for subtraction.");
    }
    matrix<DATA> m = m1;
    m -= m2;
    return m;
}
template<typename DATA>
matrix<DATA> operator-(const matrix<DATA>& m1, const double value) {
    int Size = m1.rows() * m1.cols();
    matrix<DATA> resMat = m1;

    #ifdef __AVX__
    if constexpr (std::is_same_v<DATA, float>) {
        for(int i=0; i<Size; i+=8) {
            __m256 vec = _mm256_loadu_ps(&resMat(i/resMat.cols(), i%resMat.cols()));
            __m256 valueVec = _mm256_set1_ps(static_cast<float>(value));
            vec = _mm256_sub_ps(vec, valueVec);
            _mm256_storeu_ps(&resMat(i/resMat.cols(), i%resMat.cols()), vec);
        }
    } else if constexpr (std::is_same_v<DATA, double>) {
        for(int i=0; i<Size; i+=4) {
            __m256d vec = _mm256_loadu_pd(&resMat(i/resMat.cols(), i%resMat.cols()));
            __m256d valueVec = _mm256_set1_pd(value);
            vec = _mm256_sub_pd(vec, valueVec);
            _mm256_storeu_pd(&resMat(i/resMat.cols(), i%resMat.cols()), vec);
        }
    } else {
        for(int i=0; i<Size; ++i) {
            DATA temp = resMat(i/resMat.cols(), i%resMat.cols());
            temp -= value;
            resMat(i/resMat.cols(), i%resMat.cols()) = temp;
        }
    }
    #else //if intrinsics not present
        for(int i=0; i<Size; ++i) {
            DATA temp = resMat(i/resMat.cols(), i%resMat.cols());
            temp -= value;
            resMat(i/resMat.cols(), i%resMat.cols()) = temp;
        }
    #endif
    return resMat;
}
template<typename DATA>
matrix<DATA> operator-(const double value, const matrix<DATA>& m2) {
    return m2-value;
}

/// MATRIX MULTIPLICATION
template<typename DATA>
matrix<DATA> normmatmul(const matrix<DATA> &m1,const matrix<DATA> &m2) {
    if(m1.cols() != m2.rows()) {
        throw std::invalid_argument("linear::matmul - Internal dimensions do not match.");   
    }
    matrix<DATA> m(m1.rows(), m2.cols(), DATA(0.));

    for(int i=0; i<m1.rows(); ++i) {
        for(int k=0; k<m2.rows(); ++k) {
            for(int j=0; j<m2.cols(); ++j)
                    m(i,j) += m1(i,k) * m2(k,j);  
        } // k-loop
    } //i-loop
    return m;
} 
template<typename DATA>
matrix<DATA> matmul(const matrix<DATA>& m1, const matrix<DATA>& m2) {
    return m1&m2;
}

template<typename DATA>
matrix<DATA> matmul_block(const matrix<DATA>& m1, const matrix<DATA>& m2, const int block_size=32) {
    if (m1.cols() != m2.rows()) {
        throw std::invalid_argument("linear::matmul_block - Internal dimensions do not match.");
    }

    const int m = m1.rows();
    const int n = m1.cols();
    const int p = m2.cols();

    matrix<DATA> result(m, p, (DATA)0);
    #pragma omp parallel for collapse(2) shared(m1, m2, result)
    for (int i = 0; i < m; i += block_size) {
        for (int k = 0; k < p; k += block_size) {
            for (int j = 0; j < n; j += block_size) {
                // Compute block multiplication
                for (int ii = i; ii < std::min(i + block_size, m); ++ii) {
                    for (int kk = k; kk < std::min(k + block_size, p); ++kk) {
                        DATA sum = 0;
                        #pragma omp simd
                        for (int jj = j; jj < std::min(j + block_size, n); ++jj) {
                            sum += m1(ii, jj) * m2(jj, kk);
                        }
                        result(ii, kk) += sum;
                    }
                }
            }
        }
    }

    return result;
}

#if GCCAVAIL
template<typename DATA>
matrix<DATA> matmul_blas(const matrix<DATA>& A, const matrix<DATA>& B) {
    if(A.cols() != B.rows())
        throw std::invalid_argument("linear::matmul_blas - Internal dimensions do not match.");
    
    int m = A.rows();
    int n = B.cols();
    int k = A.cols();

    matrix<DATA> C(m, n);

    // Call BLAS function for matrix-matrix multiplication
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A.begin(), k, B.begin(), n, 0.0, C.begin(), n);

    return C;
}
#endif

matrix<double> matmul_simd(const matrix<double>& A, const matrix<double>& B) {
    /*
        Utilizes SIMD instructions to perform matrix multiplication.
        In short benchmark, the results of OpenMP parallelization
        in matrix multiplication operator `&` or `matmul` are much
        faster than any implementation so far.
    */
    #if __AVX__
        if (A.cols() != B.rows()) {
            throw std::invalid_argument("linear::matmul_simd - Internal dimensions do not match.");
        }
        int rowsA = A.rows();
        int colsA = A.cols();
        int colsB = B.cols();
        
        matrix<double> result(rowsA, colsB);
        #pragma omp parallel for collapse(2) shared(A,B,result) if(rowsA >= 64 || colsA >= 64 || colsB >= 64)
        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < colsB; ++j) {
                __m256d sum = _mm256_setzero_pd();  // Initialize sum vector with zeros
                
                for (int k = 0; k < colsA; k+=8) {
                    
                    __m256d a1 = _mm256_set_pd(A(i, k + 3), A(i, k + 2), A(i, k + 1), A(i, k));
                    __m256d b1 = _mm256_set_pd(B(k + 3, j), B(k + 2, j), B(k + 1, j), B(k, j));

                    __m256d a2 = _mm256_set_pd(A(i, k + 3 + 4), A(i, k + 2 + 4), A(i, k + 1 + 4), A(i, k + 4));
                    __m256d b2 = _mm256_set_pd(B(k + 3 + 4, j), B(k + 2 + 4, j), B(k + 1 + 4, j), B(k + 4, j));

                    // Perform SIMD multiplication and addition
                    sum = _mm256_add_pd(sum, _mm256_mul_pd(a1, b1));
                    sum = _mm256_add_pd(sum, _mm256_mul_pd(a2, b2));
                }
                alignas(32) double temp[4];
                _mm256_store_pd(temp, sum);
                result(i, j) = temp[0] + temp[1] + temp[2] + temp[3];
                }
            }
        return result;
    #else
        throw std::runtime_error("linear::matmul_simd - SIMD instructions never compiled. See if you used `-mavx` flag.");
    #endif
}

// Strassen's algorithm for multiplying square matrices
template<typename DATA>
matrix<DATA> strassen_multiply(matrix<DATA> A, matrix<DATA> B, const int base_case_cutoff=32) {
    int n = A.rows();
    if (A.cols() != B.rows() || A.cols() != A.rows() || B.cols() != B.rows() || n % 2 != 0) {
        throw std::invalid_argument("Invalid matrix dimensions for Strassen's algorithm.");
    }
    
    if (n < base_case_cutoff) {
        return A&B;
    }

    int newSize = n / 2;

     // Allocate memory buffers for intermediate results
    matrix<DATA> P1(newSize, newSize);
    matrix<DATA> P2(newSize, newSize);
    matrix<DATA> P3(newSize, newSize);
    matrix<DATA> P4(newSize, newSize);
    matrix<DATA> P5(newSize, newSize);
    matrix<DATA> P6(newSize, newSize);
    matrix<DATA> P7(newSize, newSize);

    // Partition matrices
    matrix<DATA> A11 = A.slice(0, newSize, 0, newSize);
    matrix<DATA> A12 = A.slice(0, newSize, newSize, n);
    matrix<DATA> A21 = A.slice(newSize, n, 0, newSize);
    matrix<DATA> A22 = A.slice(newSize, n, newSize, n);

    matrix<DATA> B11 = B.slice(0, newSize, 0, newSize);
    matrix<DATA> B12 = B.slice(0, newSize, newSize, n);
    matrix<DATA> B21 = B.slice(newSize, n, 0, newSize);
    matrix<DATA> B22 = B.slice(newSize, n, newSize, n);

    // Compute intermediate matrices recursively
    P1 = strassen_multiply(A11 + A22, B11 + B22);
    P2 = strassen_multiply(A21 + A22, B11);
    P3 = strassen_multiply(A11, B12 - B22);
    P4 = strassen_multiply(A22, B21 - B11);
    P5 = strassen_multiply(A11 + A12, B22);
    P6 = strassen_multiply(A21 - A11, B11 + B12);
    P7 = strassen_multiply(A12 - A22, B21 + B22);

    // Calculate the submatrices of the result
    matrix<DATA> C11 = P1 + P4 - P5 + P7;
    matrix<DATA> C12 = P3 + P5;
    matrix<DATA> C21 = P2 + P4;
    matrix<DATA> C22 = P1 - P2 + P3 + P6;

    // Combine the submatrices to get the final result
    matrix<DATA> result(n, n);
    result.setSubMatrix(0, newSize, 0, newSize, C11);
    result.setSubMatrix(0, newSize, newSize, n, C12);
    result.setSubMatrix(newSize, n, 0, newSize, C21);
    result.setSubMatrix(newSize, n, newSize, n, C22);

    // Return the final result
    return result;
}

// Parallelized Strassen's algorithm for multiplying square matrices
template<typename DATA>
matrix<DATA> para_strassen_multiply(matrix<DATA> A, matrix<DATA> B, const int base_case_cutoff=32) {
    int n = A.rows();
    if (A.cols() != B.rows() || A.cols() != A.rows() || B.cols() != B.rows() || n % 2 != 0) {
        throw std::invalid_argument("Invalid matrix dimensions for Strassen's algorithm.");
    }
    
    if (n < base_case_cutoff) {
        return A&B;
    }

    int newSize = n / 2;

    // Partition matrices
    matrix<DATA> A11 = A.slice(0, newSize, 0, newSize);
    matrix<DATA> A12 = A.slice(0, newSize, newSize, n);
    matrix<DATA> A21 = A.slice(newSize, n, 0, newSize);
    matrix<DATA> A22 = A.slice(newSize, n, newSize, n);

    matrix<DATA> B11 = B.slice(0, newSize, 0, newSize);
    matrix<DATA> B12 = B.slice(0, newSize, newSize, n);
    matrix<DATA> B21 = B.slice(newSize, n, 0, newSize);
    matrix<DATA> B22 = B.slice(newSize, n, newSize, n);


    // Declare variables outside parallel sections
    matrix<DATA> P1, P2, P3, P4, P5, P6, P7;
    
    // Recursive calls
    #pragma omp parallel
    {
         #pragma omp single nowait
        {
        #pragma omp task
       /* matrix<DATA*/ P1 = strassen_multiply(A11 + A22, B11 + B22);  
        #pragma omp task 
        /* matrix<DATA*/ P2 = strassen_multiply(A21 + A22, B11);
        #pragma omp task
        /* matrix<DATA*/ P3 = strassen_multiply(A11, B12 - B22);
        #pragma omp task
        /* matrix<DATA*/ P4 = strassen_multiply(A22, B21 - B11);
        #pragma omp task
       /* matrix<DATA*/ P5 = strassen_multiply(A11 + A12, B22);
        #pragma omp task
        /* matrix<DATA*/ P6 = strassen_multiply(A21 - A11, B11 + B12);
        #pragma omp task
        /* matrix<DATA*/P7 = strassen_multiply(A12 - A22, B21 + B22);
        }
    }

    // Calculate the submatrices of the result
    matrix<DATA> C11 = P1 + P4 - P5 + P7;
    matrix<DATA> C12 = P3 + P5;
    matrix<DATA> C21 = P2 + P4;
    matrix<DATA> C22 = P1 - P2 + P3 + P6;

    // Combine the submatrices to get the final result
    matrix<DATA> result(n, n);
    result.setSubMatrix(0, newSize, 0, newSize, C11);
    result.setSubMatrix(0, newSize, newSize, n, C12);
    result.setSubMatrix(newSize, n, 0, newSize, C21);
    result.setSubMatrix(newSize, n, newSize, n, C22);

    return result;
}

/// TRIANGULAR MATRIX GENERATORS ///
matrix<double> upper_triangle_matrix(int size, double mean, double std) {
    if(size < 1)
        throw std::invalid_argument("size is less than 1.");
    
    matrix<double> result(size,size,0.);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(mean, std);
    
    for(int i=0; i<size; ++i) {
        for(int j=i; j<size; ++j) {
            result(i,j) = distribution(gen);
        }
    }

    return result;
}
matrix<double> utm(int size, double mean, double std) {
    return upper_triangle_matrix(size, mean, std);
}
matrix<double> triu(int size, double mean, double std) {
    return upper_triangle_matrix(size, mean, std);
}

matrix<double> lower_triangle_matrix(int size, double mean, double std) {
    if(size<1)
        throw std::invalid_argument("size is less than 1.");

    matrix<double> result(size,size,0.);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(mean, std);
    
    for(int i=0; i<size; ++i) {
        for(int j=0; j<=i && j<size; ++j) {
            result(i,j) = distribution(gen);
        }
    }
    return result;
}
matrix<double> ltm(int size, double mean, double std) {
    return lower_triangle_matrix(size, mean, std);
}
matrix<double> tril(int size, double mean, double std) {
    return lower_triangle_matrix(size, mean, std);
}

template<typename DATA>
void init2dArray(DATA *array, int size_0, int size_1) {
    /*
        UTIL FUNCTION
        Flattened 2d array in row major form will be initialised using this
    */
   std::cout<<"\nPlease insert "<<size_0*size_1<<" values in row major form for a "<<size_0<<'x'<<size_1<<" matrix:-\n";
    for(int i=0; i<size_0; ++i)
        for(int j=0; j<size_1; ++j)
                std::cin>>*(array + i*size_1 + j);
}

template<typename DATA>
void init2dArray(std::complex<DATA> *array, int size_0, int size_1) {
    std::cout<<"\nPlease insert "<<size_0*size_1<<" complex values in row major form for a "<<size_0<<'x'<<size_1<<" matrix:-\n";
    for(int i=0; i<size_0; ++i)
        for(int j=0; j<size_1; ++j) {
                DATA real, img;
                std::cout<<"\n\nenter real: "; std::cin>>real;
                std::cout<<"enter imag: "; std::cin>>img;
                std::complex<DATA> elem(real,img);
                *(array +i*size_1 + j) = elem;
        }
}

void init2dRandArray(int *array, int size_0, int size_1, int start=0, int end=9) {
    /*
     UTIL FUNCTION
        Flattened 2d array in row major form will be initialised using a
        uniform integer distribution.
    */ 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(start, end);

    for (int i = 0; i < size_0; ++i) {
        for (int j = 0; j < size_1; ++j)
            *(array + i * size_1 + j) = distribution(gen);
    }
}

void init2dRandArray(float *array, int size_0, int size_1, float start=0., float end=1.) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution(start, end);

    for (int i = 0; i < size_0; ++i) {
        for (int j = 0; j < size_1; ++j)
            *(array + i * size_1 + j) = distribution(gen);
    }
}

void init2dRandArray(double *array, int size_0, int size_1, double start=0., double end=1.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(start, end);

    for (int i = 0; i < size_0; ++i) {
        for (int j = 0; j < size_1; ++j)
            *(array + i * size_1 + j) = distribution(gen);
    }
}

void init2dRandArray(std::complex<double> *array, int size_0, int size_1, double start=-1., double end=1.) {
    std::complex<double> randomComplexNumber;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> realDisbn(start, end);
    std::uniform_real_distribution<double> imagDisbn(start, end);
    for (int i = 0; i < size_0; ++i) {
        for (int j = 0; j < size_1; ++j) {
            double realPart = realDisbn(gen);
            double imagPart = imagDisbn(gen);
            randomComplexNumber.real(realPart);
            randomComplexNumber.imag(imagPart);
            *(array + i * size_1 + j) = randomComplexNumber;
        }
    }
}

/*  
    GENERAL MATRIX MULTIPLY IMPLEMENTATION (GEMM) 
    References provided at the end
*/
//-- malloc with alignment --------------------------------------------------------
void *
malloc_(std::size_t alignment, std::size_t size)
{
    alignment = std::max(alignment, alignof(void *));
    size     += alignment;

    void *ptr  = std::malloc(size);
    void *ptr2 = (void *)(((uintptr_t)ptr + alignment) & ~(alignment-1));
    void **vp  = (void**) ptr2 - 1;
    *vp        = ptr;
    return ptr2;
}

void
free_(void *ptr)
{
    std::free(*((void**)ptr-1));
}

//-- Config --------------------------------------------------------------------

// SIMD-Register width in bits
// SSE:         128
// AVX/FMA:     256
// AVX-512:     512
#ifndef SIMD_REGISTER_WIDTH
#define SIMD_REGISTER_WIDTH 256
#endif

#ifdef HAVE_FMA

#   ifndef BS_D_MR
#   define BS_D_MR 4
#   endif

#   ifndef BS_D_NR
#   define BS_D_NR 12
#   endif

#   ifndef BS_D_MC
#   define BS_D_MC 256
#   endif

#   ifndef BS_D_KC
#   define BS_D_KC 512
#   endif

#   ifndef BS_D_NC
#   define BS_D_NC 4092
#   endif

#endif



#ifndef BS_D_MR
#define BS_D_MR 4
#endif

#ifndef BS_D_NR
#define BS_D_NR 8
#endif

#ifndef BS_D_MC
#define BS_D_MC 256
#endif

#ifndef BS_D_KC
#define BS_D_KC 256
#endif

#ifndef BS_D_NC
#define BS_D_NC 4096
#endif

template <typename T>
struct BlockSize
{
    static constexpr int MC = 64;
    static constexpr int KC = 64;
    static constexpr int NC = 256;
    static constexpr int MR = 8;
    static constexpr int NR = 8;

    static constexpr int rwidth = 0;
    static constexpr int align  = alignof(T);
    static constexpr int vlen   = 0;

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0, "Invalid block size.");
    static_assert(MC % MR == 0, "MC must be a multiple of MR.");
    static_assert(NC % NR == 0, "NC must be a multiple of NR.");
};


template <>
struct BlockSize<double>
{
    static constexpr int MC     = BS_D_MC;
    static constexpr int KC     = BS_D_KC;
    static constexpr int NC     = BS_D_NC;
    static constexpr int MR     = BS_D_MR;
    static constexpr int NR     = BS_D_NR;

    static constexpr int rwidth = SIMD_REGISTER_WIDTH;
    static constexpr int align  = rwidth / 8;
    static constexpr int vlen   = rwidth / (8*sizeof(double));

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0, "Invalid block size.");
    static_assert(MC % MR == 0, "MC must be a multiple of MR.");
    static_assert(NC % NR == 0, "NC must be a multiple of NR.");
    static_assert(rwidth % sizeof(double) == 0, "SIMD register width not sane.");
};

template <>
struct BlockSize<int>
{
    static constexpr int MC     = BS_D_MC;
    static constexpr int KC     = BS_D_KC;
    static constexpr int NC     = BS_D_NC;
    static constexpr int MR     = BS_D_MR;
    static constexpr int NR     = BS_D_NR;

    static constexpr int rwidth = SIMD_REGISTER_WIDTH;
    static constexpr int align  = rwidth / 8;
    static constexpr int vlen   = rwidth / (8*sizeof(int));

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0, "Invalid block size.");
    static_assert(MC % MR == 0, "MC must be a multiple of MR.");
    static_assert(NC % NR == 0, "NC must be a multiple of NR.");
    static_assert(rwidth % sizeof(int) == 0, "SIMD register width not sane.");
};

template <>
struct BlockSize<float>
{
    static constexpr int MC     = BS_D_MC;
    static constexpr int KC     = BS_D_KC;
    static constexpr int NC     = BS_D_NC;
    static constexpr int MR     = BS_D_MR;
    static constexpr int NR     = BS_D_NR;

    static constexpr int rwidth = SIMD_REGISTER_WIDTH;
    static constexpr int align  = rwidth / 8;
    static constexpr int vlen   = rwidth / (8*sizeof(float));

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0, "Invalid block size.");
    static_assert(MC % MR == 0, "MC must be a multiple of MR.");
    static_assert(NC % NR == 0, "NC must be a multiple of NR.");
    static_assert(rwidth % sizeof(float) == 0, "SIMD register width not sane.");
};

template <>
struct BlockSize<long>
{
    static constexpr int MC     = BS_D_MC;
    static constexpr int KC     = BS_D_KC;
    static constexpr int NC     = BS_D_NC;
    static constexpr int MR     = BS_D_MR;
    static constexpr int NR     = BS_D_NR;

    static constexpr int rwidth = SIMD_REGISTER_WIDTH;
    static constexpr int align  = rwidth / 8;
    static constexpr int vlen   = rwidth / (8*sizeof(long));

    static_assert(MC>0 && KC>0 && NC>0 && MR>0 && NR>0, "Invalid block size.");
    static_assert(MC % MR == 0, "MC must be a multiple of MR.");
    static_assert(NC % NR == 0, "NC must be a multiple of NR.");
    static_assert(rwidth % sizeof(long) == 0, "SIMD register width not sane.");
};

//-- aux routines --------------------------------------------------------------
template <typename Index, typename Alpha, typename TX, typename TY>
void
geaxpy(Index m, Index n,
       const Alpha &alpha,
       const TX *X, Index incRowX, Index incColX,
       TY       *Y, Index incRowY, Index incColY)
{
    for (Index j=0; j<n; ++j) {
        for (Index i=0; i<m; ++i) {
            Y[i*incRowY+j*incColY] += alpha*X[i*incRowX+j*incColX];
        }
    }
}

template <typename Index, typename Alpha, typename TX>
void
gescal(Index m, Index n,
       const Alpha &alpha,
       TX *X, Index incRowX, Index incColX)
{
    if (alpha!=Alpha(0)) {
        for (Index j=0; j<n; ++j) {
            for (Index i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] *= alpha;
            }
        }
    } else {
        for (Index j=0; j<n; ++j) {
            for (Index i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] = Alpha(0);
            }
        }
    }
}

//-- Micro Kernel --------------------------------------------------------------
template <typename Index, typename T>
typename std::enable_if<BlockSize<T>::vlen != 0,
         void>::type
ugemm(Index kc, T alpha, const T *A, const T *B, T beta,
      T *C, Index incRowC, Index incColC)
{
    typedef T vx __attribute__((vector_size (BlockSize<T>::rwidth/8)));

    static constexpr Index vlen = BlockSize<T>::vlen;
    static constexpr Index MR   = BlockSize<T>::MR;
    static constexpr Index NR   = BlockSize<T>::NR/vlen;

    A = (const T*) __builtin_assume_aligned (A, BlockSize<T>::align);
    B = (const T*) __builtin_assume_aligned (B, BlockSize<T>::align);

    vx P[MR*NR] = {};

    for (Index l=0; l<kc; ++l) {
        const vx *b = (const vx *)B;
        for (Index i=0; i<MR; ++i) {
            for (Index j=0; j<NR; ++j) {
                P[i*NR+j] += A[i]*b[j];
            }
        }
        A += MR;
        B += vlen*NR;
    }

    if (alpha!=T(1)) {
        for (Index i=0; i<MR; ++i) {
            for (Index j=0; j<NR; ++j) {
                P[i*NR+j] *= alpha;
            }
        }
    }

    if (beta!=T(0)) {
        for (Index i=0; i<MR; ++i) {
            for (Index j=0; j<NR; ++j) {
                const T *p = (const T *) &P[i*NR+j];
                for (Index j1=0; j1<vlen; ++j1) {
                    C[i*incRowC+(j*vlen+j1)*incColC] *= beta;
                    C[i*incRowC+(j*vlen+j1)*incColC] += p[j1];
                }
            }
        }
    } else {
        for (Index i=0; i<MR; ++i) {
            for (Index j=0; j<NR; ++j) {
                const T *p = (const T *) &P[i*NR+j];
                for (Index j1=0; j1<vlen; ++j1) {
                    C[i*incRowC+(j*vlen+j1)*incColC] = p[j1];
                }
            }
        }
    }
}

//-- Macro Kernel --------------------------------------------------------------
template <typename Index, typename T, typename Beta, typename TC>
void
mgemm(Index mc, Index nc, Index kc,
      T alpha,
      const T *A, const T *B,
      Beta beta,
      TC *C, Index incRowC, Index incColC)
{
    const Index MR = BlockSize<T>::MR;
    const Index NR = BlockSize<T>::NR;
    const Index mp  = (mc+MR-1) / MR;
    const Index np  = (nc+NR-1) / NR;
    const Index mr_ = mc % MR;
    const Index nr_ = nc % NR;

    T C_[MR*NR];

    #pragma omp parallel for
    for (Index j=0; j<np; ++j) {
        const Index nr = (j!=np-1 || nr_==0) ? NR : nr_;

        for (Index i=0; i<mp; ++i) {
            const Index mr = (i!=mp-1 || mr_==0) ? MR : mr_;

            if (mr==MR && nr==NR) {
                ugemm(kc, alpha,
                      &A[i*kc*MR], &B[j*kc*NR],
                      beta,
                      &C[i*MR*incRowC+j*NR*incColC],
                      incRowC, incColC);
            } else {
                ugemm(kc, alpha,
                      &A[i*kc*MR], &B[j*kc*NR],
                      T(0),
                      C_, Index(1), MR);
                gescal(mr, nr, beta,
                       &C[i*MR*incRowC+j*NR*incColC],
                       incRowC, incColC);
                geaxpy(mr, nr, T(1), C_, Index(1), MR,
                       &C[i*MR*incRowC+j*NR*incColC],
                       incRowC, incColC);
            }
        }
    }
}
//-- Packing blocks ------------------------------------------------------------
template <typename Index, typename TA, typename T>
void
pack_A(Index mc, Index kc,
       const TA *A, Index incRowA, Index incColA,
       T *p)
{
    Index MR = BlockSize<T>::MR;
    Index mp = (mc+MR-1) / MR;

    for (Index j=0; j<kc; ++j) {
        for (Index l=0; l<mp; ++l) {
            for (Index i0=0; i0<MR; ++i0) {
                Index i  = l*MR + i0;
                Index nu = l*MR*kc + j*MR + i0;
                p[nu]   = (i<mc) ? A[i*incRowA+j*incColA]
                                 : T(0);
            }
        }
    }
}

template <typename Index, typename TB, typename T>
void
pack_B(Index kc, Index nc,
       const TB *B, Index incRowB, Index incColB,
       T *p)
{
    Index NR = BlockSize<T>::NR;
    Index np = (nc+NR-1) / NR;

    for (Index l=0; l<np; ++l) {
        for (Index j0=0; j0<NR; ++j0) {
            for (Index i=0; i<kc; ++i) {
                Index j  = l*NR+j0;
                Index nu = l*NR*kc + i*NR + j0;
                p[nu]   = (j<nc) ? B[i*incRowB+j*incColB]
                                 : T(0);
            }
        }
    }
}
//-- Frame routine -------------------------------------------------------------
template <typename Index, typename Alpha,
         typename TA, typename TB,
         typename Beta,
         typename TC>
void
gemm(Index m, Index n, Index k,
     Alpha alpha,
     const TA *A, Index incRowA, Index incColA,
     const TB *B, Index incRowB, Index incColB,
     Beta beta,
     TC *C, Index incRowC, Index incColC)
{
    typedef typename std::common_type<Alpha, TA, TB>::type  T;

    const Index MC = BlockSize<T>::MC;
    const Index NC = BlockSize<T>::NC;
    const Index MR = BlockSize<T>::MR;
    const Index NR = BlockSize<T>::NR;

    const Index KC = BlockSize<T>::KC;
    const Index mb = (m+MC-1) / MC;
    const Index nb = (n+NC-1) / NC;
    const Index kb = (k+KC-1) / KC;
    const Index mc_ = m % MC;
    const Index nc_ = n % NC;
    const Index kc_ = k % KC;

    T *A_ = (T*) malloc_(BlockSize<T>::align, sizeof(T)*(MC*KC+MR));
    T *B_ = (T*) malloc_(BlockSize<T>::align, sizeof(T)*(KC*NC+NR));

    if (alpha==Alpha(0) || k==0) {
        gescal(m, n, beta, C, incRowC, incColC);
        return;
    }

    for (Index j=0; j<nb; ++j) {
        Index nc = (j!=nb-1 || nc_==0) ? NC : nc_;

        for (Index l=0; l<kb; ++l) {
            Index   kc  = (l!=kb-1 || kc_==0) ? KC : kc_;
            Beta beta_  = (l==0) ? beta : Beta(1);

            pack_B(kc, nc,
                   &B[l*KC*incRowB+j*NC*incColB],
                   incRowB, incColB,
                   B_);

            for (Index i=0; i<mb; ++i) {
                Index mc = (i!=mb-1 || mc_==0) ? MC : mc_;

                pack_A(mc, kc,
                       &A[i*MC*incRowA+l*KC*incColA],
                       incRowA, incColA,
                       A_);

                mgemm(mc, nc, kc,
                      T(alpha), A_, B_, beta_,
                      &C[i*MC*incRowC+j*NC*incColC],
                      incRowC, incColC);
            }
        }
    }
    free_(A_);
    free_(B_);
}

//------------------------------------------------------------------------------
// works on the internal pointer
void matrixproduct(double *c, const double* a, const double* b, int N) {
    gemm(N, N, N, 1.0, a, N, 1, b, N, 1, 0.0, c, N, 1);
}

// GEMM implemented within linear::operator&
template<typename DATA>
matrix<DATA> operator&(const matrix<DATA>& A, const matrix<DATA>& B) {
    const int M = A.rows();
    const int K = A.cols();
    const int N = B.cols();


    if(A.cols() != B.rows())
        throw std::domain_error("linear::operator& - Internal dimensions do not match.");

    matrix<DATA> C(M,N);

    if(M <= 128 || N <= 128 || K <= 128)
        return normmatmul(A,B);

    gemm(M,N,K, (DATA)1.0, A.begin(), M, 1, B.begin(), K, 1, (DATA)0.0, C.begin(), M, 1);
    return C;
}
/* Reference for GEMM code above
Code sourced from and modified:-
https://stackoverflow.com/a/35637007

Inspired by the BLIS framework:-
https://www.cs.utexas.edu/users/flame/pubs/blis1_toms_rev3.pdf
*/

}//linear namespace ends
#endif // MATRIX_H