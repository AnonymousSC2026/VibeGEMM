#include <cstdio>

#include "engine.cuh"
#include "registry.cuh"


struct GemmProblem {
        const char* label;
        int m, n, k;
};
    
static const GemmProblem sizes[] = {
    // // ── Square sweep ──────────────────────────────────────────────────────────
    // {nullptr,  256,  256,  256},  
    // {nullptr,  512,  512,  512},
    // {nullptr,  768,  768,  768},  
    // {nullptr, 1024, 1024, 1024},
    // {nullptr, 1536, 1536, 1536},  
    // {nullptr, 2048, 2048, 2048},
    // {nullptr, 3072, 3072, 3072},  
    // {nullptr, 4096, 4096, 4096},
    // {nullptr, 5120, 5120, 5120},  
    // {nullptr, 6144, 6144, 6144},
    // {nullptr, 7168, 7168, 7168},  
    {nullptr, 8192, 8192, 8192}, // test 8192 x 8192 x 8192
    // {nullptr, 9216, 9216, 9216},  
    // {nullptr,10240,10240,10240},
    // {nullptr,11264,11264,11264},  
    // {nullptr,12288,12288,12288},
    // {nullptr,13312,13312,13312},  
    // {nullptr,14336,14336,14336},
    // {nullptr,15360,15360,15360},  
    // {nullptr,16384,16384,16384},

    // // ── Transformer / LLM shapes ──────────────────────────────────────────────
    // {"BERT-Large inference",  384,  1024,  1024},
    // {"GPT-3 training",       2048, 12288, 36864},
    // {"Llama 3.1 405B",       8192, 16384, 16384},
    // {"DeepSeek-V3",          4096,  7168,  7168},
    // {"Mixtral-8x7B",          512,  4096, 14336},

    // // ── Edge cases ────────────────────────────────────────────────────────────
    // {"HPL",         100000,  128,   128},
    // {"MoE Router",    8192, 4096,    16},
    // {"HPC Krylov",  200000,  256,    32},
};


int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm = prop.major * 10 + prop.minor;

    system("mkdir -p results");

    std::string log_file = (sm >= 90) ? "results/perf_h100.txt" : "results/perf_a100.txt";
    FILE* fp = fopen(log_file.c_str(), "w");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open %s for writing!\n", log_file.c_str());
        return 1;
    }

    printf("=== Device: %s (SM: %d) ===\n", prop.name, sm);
    fprintf(fp, "=== Device: %s (SM: %d) ===\n", prop.name, sm);

    BenchmarkEngine engine;

    for (auto& b : KernelRegistry::instance().make_all()) {
        std::string name = b->name();
        if (sm >= 90 && name.find("A100") != std::string::npos)
            continue;
        if (sm < 90 && name.find("H100") != std::string::npos)
            continue;
        engine.add_backend(b);
    }

    for (const auto& c : sizes) {
        if (c.label) {
            printf("\n>>> %s\n", c.label);
            fprintf(fp, "\n>>> %s\n", c.label);
        }
        engine.execute(c.m, c.n, c.k);
    }

    printf("\nDone.\n");
    fprintf(fp, "\nDone.\n");

    fflush(stdout);
    fclose(fp);
    return 0;
}