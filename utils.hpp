#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdlib>
#include <new>

template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator() noexcept = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        // Size must be a multiple of alignment for std::aligned_alloc
        std::size_t size = n * sizeof(T);
        std::size_t remainder = size % Alignment;
        if (remainder != 0) size += (Alignment - remainder);

        // void* ptr = std::aligned_alloc(Alignment, size);
        void* ptr = ::operator new(size, std::align_val_t(Alignment));
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        // std::free(p);
        ::operator delete(p, std::align_val_t(Alignment));
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

// Equality operators required by allocator traits
template <typename T, typename U, std::size_t Align>
bool operator==(const AlignedAllocator<T, Align>&, const AlignedAllocator<U, Align>&) { return true; }
template <typename T, typename U, std::size_t Align>
bool operator!=(const AlignedAllocator<T, Align>&, const AlignedAllocator<U, Align>&) { return false; }

float randd(float l, float r);

float sigmoid(float x);

#endif