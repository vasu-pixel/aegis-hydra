
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <omp.h>

// Fast Random Number Generator
struct XorShift128 {
    uint32_t x, y, z, w;
    XorShift128(uint32_t seed) {
        x = seed; y = 362436069; z = 521288629; w = 88675123;
    }
    inline uint32_t next() {
        uint32_t t = x ^ (x << 11);
        x = y; y = z; z = w;
        return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
    }
    inline float next_float() {
        return (next() & 0xFFFFFF) / 16777216.0f;
    }
};

class IsingModel {
public:
    int height;
    int width;
    std::vector<int8_t> spins;
    std::vector<XorShift128> rngs;

    IsingModel(int h, int w, uint32_t seed) : height(h), width(w), spins(h * w) {
        // Initialize spins randomly
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> dis(0, 1);
        for (int i = 0; i < h * w; ++i) {
            spins[i] = dis(gen) ? 1 : -1;
        }
        
        // Per-thread RNGs
        int n_threads = omp_get_max_threads();
        for (int i = 0; i < n_threads; ++i) {
            rngs.emplace_back(seed + i);
        }
    }
    
    // Checkboard Update (Red/Black)
    // 0 = Red (even sum), 1 = Black (odd sum)
    void step(float T, float J, float h) {
        update_color(0, T, J, h); // Update Red
        update_color(1, T, J, h); // Update Black
    }
    
    void update_color(int color, float T, float J, float h) {
        // Precompute exponentials for speed
        // Delta E can be -8J, -4J, 0, 4J, 8J (ignoring h)
        // We calculate exact exp inside loop as h is float. 
        // Or we could cache if h is constant? h changes every step.
        
        #pragma omp parallel for
        for (int i = 0; i < height; ++i) {
            int thread_id = omp_get_thread_num();
            // Row offset
            // We want (i + j) % 2 == color
            // If i%2 == 0, j must match color
            // If i%2 == 1, j must be !color
            int start_j = (i % 2 == color) ? 0 : 1;
            
            for (int j = start_j; j < width; j += 2) {
                int idx = i * width + j;
                int8_t sigma = spins[idx];
                
                // Neighbors (Periodic Boundary)
                int up    = (i == 0) ? (height - 1) * width + j : (i - 1) * width + j;
                int down  = (i == height - 1) ? j : (i + 1) * width + j;
                int left  = (j == 0) ? i * width + (width - 1) : i * width + (j - 1);
                int right = (j == width - 1) ? i * width : i * width + (j + 1);
                
                int sum_neighbors = spins[up] + spins[down] + spins[left] + spins[right];
                
                // Effective Field B = J * sum + h
                float B = J * sum_neighbors + h;
                
                // Energy Change DeltaE = 2 * sigma * B
                // If we flip sigma -> -sigma
                float delta_E = 2.0f * sigma * B;
                
                bool flip = false;
                if (delta_E < 0) {
                    flip = true;
                } else {
                    float p = std::exp(-delta_E / T);
                    if (rngs[thread_id].next_float() < p) {
                        flip = true;
                    }
                }
                
                if (flip) {
                    spins[idx] = -sigma;
                }
            }
        }
    }
    
    float magnetization() {
        long long sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < height * width; ++i) {
            sum += spins[i];
        }
        return (float)sum / (height * width);
    }
};

extern "C" {
    IsingModel* Ising_new(int h, int w, uint32_t seed) {
        return new IsingModel(h, w, seed);
    }
    
    void Ising_delete(IsingModel* model) {
        delete model;
    }
    
    void Ising_step(IsingModel* model, float T, float J, float h) {
        model->step(T, J, h);
    }
    
    float Ising_magnetization(IsingModel* model) {
        return model->magnetization();
    }
    
    // Get pointer to raw spins for visualization (copy to numpy)
    int8_t* Ising_get_spins(IsingModel* model) {
        return model->spins.data();
    }
}
