#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace lumina
{

__constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__constant__ uint32_t H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

//

__device__
static inline uint32_t rotr(uint32_t x, uint32_t n)
{
    return (x >> n) | (x << (32 - n));
}

template <size_t S>
__device__
static void sha256_message(
    const __restrict__ uint8_t* message,
          __restrict__ uint8_t* digest
) {
    constexpr int BLOCK_SIZE = 64; // Block size for SHA-256 (512 bits)
    static_assert(S % BLOCK_SIZE == 0);

    // Initialize hash values
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t hash[8] = {H0[0], H0[1], H0[2], H0[3], H0[4], H0[5], H0[6], H0[7]};

    // Process each 512-bit block in the padded message
    for (size_t block_offset = 0; block_offset < S; block_offset += BLOCK_SIZE) {
        // Initialize working variables
        a = hash[0];
        b = hash[1];
        c = hash[2];
        d = hash[3];
        e = hash[4];
        f = hash[5];
        g = hash[6];
        h = hash[7];

        // Prepare the message schedule array (w)
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (message[block_offset + i * 4] << 24) |
                   (message[block_offset + i * 4 + 1] << 16) |
                   (message[block_offset + i * 4 + 2] << 8) |
                   (message[block_offset + i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        // Compression function main loop
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t temp1 = h + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        // Add the compressed chunk to the current hash value
        hash[0] += a;
        hash[1] += b;
        hash[2] += c;
        hash[3] += d;
        hash[4] += e;
        hash[5] += f;
        hash[6] += g;
        hash[7] += h;
    }

    // Write the final hash value to the output buffer
    for (int i = 0; i < 8; ++i) {
        digest[i * 4] = (hash[i] >> 24) & 0xff;
        digest[i * 4 + 1] = (hash[i] >> 16) & 0xff;
        digest[i * 4 + 2] = (hash[i] >> 8) & 0xff;
        digest[i * 4 + 3] = hash[i] & 0xff;
    }
}

template <size_t S>
__global__
void sha256(
    const              size_t   nmessages, 
    const __restrict__ uint8_t* messages, 
          __restrict__ uint8_t* hashes
) {
    size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= nmessages) return;

    sha256_message<S>(messages + S * x, hashes + (32 * x));
}

}
