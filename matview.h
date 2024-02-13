#ifndef MATRIX_VIEW_H
#define MATRIX_VIEW_H

#include "matrix.h"

namespace linear{
template<typename DATA>
class matrixView {
    private:
    matrix<DATA>& orgMatrix;
    range rowRange;
    range colRange;

    public:
    matrixView(matrix<DATA>& mat, int startRow, int endRow, int startCol, int endCol):\
    orgMatrix(mat), rowRange(startRow, endRow), colRange(startCol, endCol) {}

    matrixView(matrix<DATA>& mat, range rowRng, range colRng) :\
    orgMatrix(mat), rowRange(rowRng), colRange(colRng) {}

    DATA& operator()(int r, int c) {
        if (r < rowRange.start || r >= rowRange.end || c < colRange.start || c >= colRange.end) {
            throw std::out_of_range("matrixView () - Index out of range.");
        }

        int adjustedRowIndex = r - rowRange.start;
        int adjustedColIndex = c - colRange.start;

        return orgMatrix(adjustedRowIndex, adjustedColIndex);
    }

    void display(const std::string msg=":-");
    matrix<DATA> cvtToMatrix();
};


template<typename DATA>
matrix<DATA> matrixView<DATA>::cvtToMatrix() {
    matrix<DATA> res(rowRange.length, colRange.length);
    for(int i=0; i<rowRange.length; ++i) {
        for(int j=0; j<colRange.length; ++j) {
            res(i,j) = orgMatrix(rowRange.start + i, colRange.start +j);
        }
    }

    return res;
}

template<typename DATA>
void matrixView<DATA>::display(const std::string msg) {
    //experimental code
    int i,j;
    std::cout<<"\n[(matrixView)] "<<msg<<"\n";

    // zero size matrix display
    if(this->rowRange.size() == 0 || this->colRange.size() == 0) {
        std::cout<<"(empty matrix)\n";
        return;
    }

    int max_precision = MATRIX_PRECISION;
    int padding = 1;

    // Find the maximum number of digits in the matrix
    int maxDigits = 1;
    for (i = 0; i < this->rowRange.size(); ++i) {
        for (j = 0; j < this->colRange.size(); ++j) {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(max_precision) << orgMatrix(rowRange.start + i, colRange.start + j);
            std::string str = stream.str();

            size_t pos = str.find_last_not_of('0');
            if (pos != std::string::npos && str[pos] == '.')
                pos--;

            maxDigits = std::max(maxDigits, static_cast<int>(pos + 1));
        }
    }

    // Set the width based on the maximum number of digits
    int width = maxDigits + padding;
    for (i = 0; i < this->rowRange.size(); ++i) {
        for (j = 0; j < this->colRange.size(); ++j) {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(max_precision) << orgMatrix(rowRange.start + i, colRange.start + j);
            std::string str = stream.str();

            size_t pos = str.find_last_not_of('0');
            if (pos != std::string::npos && str[pos] == '.')
                pos--;
            std::cout << std::setw(width) << str.substr(0, pos + 1);
        }
        std::cout << "\n";
    }
}

}//linear namepsace

#endif