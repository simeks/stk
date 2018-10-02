#pragma once

#include <stk/common/assert.h>
#include <stk/math/float3.h>

#include <algorithm>
#include <initializer_list>

struct Matrix3x3f
{
    static constexpr unsigned int rows = 3;
    static constexpr unsigned int cols = 3;

    float3 _rows[rows];

    float3& operator[](const unsigned int i) {
        return _rows[i];
    }

    const float3& operator[](const unsigned int i) const {
        return _rows[i];
    }

    const float& operator()(const unsigned int r, const unsigned int c) const {
        ASSERT(c < 3 && r < 3);
        return *(reinterpret_cast<const float*>(_rows + r) + c);
    }

    float& operator()(const unsigned int r, const unsigned int c) {
        return const_cast<float&>(static_cast<const Matrix3x3f*>(this)->operator()(r, c));
    }

    float* data(void) {
        return reinterpret_cast<float*>(&_rows[0]);
    }

    const float* data(void) const {
        return reinterpret_cast<const float*>(&_rows[0]);
    }

    void set(const float *data) {
        std::copy(data, data + rows*cols, (float*) _rows);
    }

    void set(const std::initializer_list<float> data) {
        std::copy(data.begin(), data.end(), (float*) _rows);
    }

    void diagonal(const std::initializer_list<float> d) {
        ASSERT(d.size() == std::min(rows, cols));
        std::fill(data(), data() + rows * cols, float(0));
        int i = 0;
        for (const auto& x : d) {
            (*this)(i, i) = x;
            i++;
        }
    }

    float determinant(void) const {
        return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1)) -
               (*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0)) +
               (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
    }

    Matrix3x3f inverse(void) const {
        const float det = determinant();
        if (std::abs(det) < std::numeric_limits<float>::epsilon()) {
            FATAL() << "The matrix is not invertible";
        }
        // NOTE: we cannot assume that the direction matrix is orthogonal
        const float inv_det = 1.0 / det;
        Matrix3x3f res;
        res(0, 0) = inv_det * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1));
        res(0, 1) = inv_det * ((*this)(0, 2) * (*this)(2, 1) - (*this)(0, 1) * (*this)(2, 2));
        res(0, 2) = inv_det * ((*this)(0, 1) * (*this)(1, 2) - (*this)(0, 2) * (*this)(1, 1));
        res(1, 0) = inv_det * ((*this)(1, 2) * (*this)(2, 0) - (*this)(1, 0) * (*this)(2, 2));
        res(1, 1) = inv_det * ((*this)(0, 0) * (*this)(2, 2) - (*this)(0, 2) * (*this)(2, 0));
        res(1, 2) = inv_det * ((*this)(0, 2) * (*this)(1, 0) - (*this)(0, 0) * (*this)(1, 2));
        res(2, 0) = inv_det * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
        res(2, 1) = inv_det * ((*this)(0, 1) * (*this)(2, 0) - (*this)(0, 0) * (*this)(2, 1));
        res(2, 2) = inv_det * ((*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0));
        return res;
    }

    const float3 operator*(const float3& right) const {
        return float3({
            dot(_rows[0], right),
            dot(_rows[1], right),
            dot(_rows[2], right)
        });
    }
};

