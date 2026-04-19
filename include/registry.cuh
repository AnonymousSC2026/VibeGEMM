#pragma once
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "backend.cuh"

class KernelRegistry {
  public:
    using Factory = std::function<std::shared_ptr<GemmBackend>()>;

    static KernelRegistry &instance() {
        static KernelRegistry reg;
        return reg;
    }

    void add(int priority, const char *display_name, Factory f) {
        entries_.push_back({priority, display_name, std::move(f)});
    }

    std::vector<std::shared_ptr<GemmBackend>> make_all() const {
        auto sorted = entries_;
        std::stable_sort(sorted.begin(), sorted.end(),
                         [](const Entry &a, const Entry &b) { return a.priority < b.priority; });
        std::vector<std::shared_ptr<GemmBackend>> out;
        out.reserve(sorted.size());
        for (const auto &e : sorted)
            out.push_back(e.factory());
        return out;
    }

  private:
    KernelRegistry() = default;
    struct Entry {
        int priority;
        const char *name;
        Factory factory;
    };
    std::vector<Entry> entries_;
};

// ── KernelRegistrar ───────────────────────────────────────────────────────────
struct KernelRegistrar {
    KernelRegistrar(int priority, const char *name, KernelRegistry::Factory f) {
        KernelRegistry::instance().add(priority, name, std::move(f));
    }
};

#define _KREG_CONCAT_INNER(a, b) a##b
#define _KREG_CONCAT(a, b) _KREG_CONCAT_INNER(a, b)

#define _KREG_IMPL(priority, dn, BC, N)                                 \
    static ::KernelRegistrar _KREG_CONCAT(_kreg_, N)(                   \
        priority, dn,                                                   \
        []() -> std::shared_ptr<GemmBackend> {                          \
            return std::make_shared<BC>(dn);                            \
        })

#define _KREG_IMPL_NOARG(priority, dn, BC, N)                           \
    static ::KernelRegistrar _KREG_CONCAT(_kreg_, N)(                   \
        priority, dn,                                                   \
        []() -> std::shared_ptr<GemmBackend> {                          \
            return std::make_shared<BC>();                              \
        })

#define REGISTER_CUBLAS(dn, BC)       _KREG_IMPL_NOARG(0, dn, BC, __COUNTER__)
#define REGISTER_A100_KERNEL(dn, BC)  _KREG_IMPL(1, dn, BC, __COUNTER__)
#define REGISTER_H100_KERNEL(dn, BC)  _KREG_IMPL(2, dn, BC, __COUNTER__)
