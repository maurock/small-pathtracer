#ifndef UTILITIES
#define UTILITIES

/*
 * utilities.cpp
 *
 *  Created on: Apr 26, 2019
 *      Author: mauro
 */

#include "math.h"   // smallpt, a Path Tracer by Kevin Beason, 2008

#define RAND48_MULT_0   (0xe66d)
#define RAND48_MULT_1   (0xdeec)
#define RAND48_MULT_2   (0x0005)
#define RAND48_ADD      (0x000b)


unsigned short _rand48_add = RAND48_ADD;
unsigned short _rand48_mult[3] = {
    RAND48_MULT_0,
    RAND48_MULT_1,
    RAND48_MULT_2
};

void _dorand48(unsigned short xseed[3])
{
    unsigned long accu;
    unsigned short temp[2];

    accu = (unsigned long)_rand48_mult[0] * (unsigned long)xseed[0] +
        (unsigned long)_rand48_add;
    temp[0] = (unsigned short)accu;        /* lower 16 bits */
    accu >>= sizeof(unsigned short) * 8;
    accu += (unsigned long)_rand48_mult[0] * (unsigned long)xseed[1] +
        (unsigned long)_rand48_mult[1] * (unsigned long)xseed[0];
    temp[1] = (unsigned short)accu;        /* middle 16 bits */
    accu >>= sizeof(unsigned short) * 8;
    accu += _rand48_mult[0] * xseed[2] + _rand48_mult[1] * xseed[1] + _rand48_mult[2] * xseed[0];
    xseed[0] = temp[0];
    xseed[1] = temp[1];
    xseed[2] = (unsigned short)accu;
}

float erand48(unsigned short xseed[3])
{
    _dorand48(xseed);
    return ldexp((float)xseed[0], -48) +
        ldexp((float)xseed[1], -32) +
        ldexp((float)xseed[2], -16);
}

// operators for std::array<>
template<typename T, std::size_t N>
std::array<T, N> operator*(std::array<T, N>  array, float n) {    // operator* for std::array
	std::array<T, N> temp;
	for(int i = 0; i < n; i++){
		temp[i] = array[i]*n;
	}
	return temp;
}

template<typename T, std::size_t N>
std::array<T, N> operator+(std::array<T, N>  array1, std::array<T, N>  array2) {    // operator+ for std::array
	std::array<T, N> temp;
	for(int i = 0; i < array1.size(); i++){
		temp[i] = array1[i]+ array2[i];
	}
	return temp;
}

#endif
