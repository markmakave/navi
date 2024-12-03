#pragma once

#include "cuda/u256.cuh"

namespace lumina::ecdsa
{

// __constant__ static u256 GX = { 0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07, 0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798 };
// __constant__ static u256 GY = { 0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8, 0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8 };

static constexpr u64 P_raw[] = { 0xFFFFFC2FFFFFFFFE, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
__constant__ static u256 P;

__device__
u256 gcd(u256 lhs, u256 rhs)
{
    if (lhs == 0) return rhs;
    if (rhs == 0) return lhs;

    u32 s = 0;

    // Factor out common powers of 2
    while ((lhs & 1) == 0 && (rhs & 1) == 0)
    {
        lhs >>= 1;
        rhs >>= 1;
        s++;
    }

    while (rhs != 0)
    {
        while ((lhs & 1) == 0)
            lhs >>= 1;

        while ((rhs & 1) == 0)
            rhs >>= 1;

        if (lhs > rhs)
            lhs -= rhs;
        else
            rhs -= lhs;
    }

    return lhs << s;
}

__device__ __forceinline__
u256 barrett_reduction(const u256& x, const u256& m, const u256& mu)
{

}


__device__ __forceinline__
static u256 inverse_mod(const u256& x, const u256& m)
{
    
}

__device__ __forceinline__
static u256 montgomery_mul(const u256& x, const u256& y, const u256& m)
{
    
    return m;
}

__device__ __forceinline__
static u256 add_modulo(const u256& lhs, const u256& rhs, const u256& m)
{
    assert(lhs < m);
    assert(rhs < m);
    
    u256 result = lhs + rhs;
    if (result < lhs)
        result -= m;
    else
        result += m;

    return result;
}

__device__ __forceinline__
static u256 sub_modulo(const u256& lhs, const u256& rhs, const u256& m)
{
    assert(lhs < m);
    assert(rhs < m);
    
    u256 result = lhs - rhs;
    if (lhs < rhs)
        result += m;
    else
        result -= m;

    return result;
}

__device__
static u256 euclidean_division(const u256& lhs, const u256& rhs, const u256& m)
{
    return lhs * inverse_mod(rhs, m) % m;
}

struct point
{
    u256 x, y;

    __device__ __forceinline__
    point operator+ (const point& rhs) const
    {
        u256 slope = euclidean_division((y - rhs.y), (x - rhs.x), P);
        u256 x_r = slope * slope - x - rhs.x;
        u256 y_r = y + slope * (x_r - x);

        return { x_r % P, -y_r % P };
    }

    __device__ __forceinline__
    point operator* (const u256& scalar)
    {

    }

};

}
