// Simply std::vector implementation C++

#include <algorithm>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <utility>

template<typename T>
class Vector {
public:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Constructors

    Vector() : size_(0), capacity_(0), buf_(nullptr) {
    }

    Vector(const Vector& other)
            : size_(other.size_)
            , capacity_(other.capacity_)
            , buf_(reinterpret_cast<T*>(new char[capacity_ * sizeof(T)])) {
        for (size_t i = 0; i < size_; ++i) {
            new (buf_ + i) T(other[i]);
        }
    }

    Vector(Vector&& other)
            : size_(other.size_)
            , capacity_(other.capacity_)
            , buf_(other.buf_) {
        other.buf_ = nullptr;
        other.capacity_ = 0;
        other.size_ = 0;
    }

    Vector(size_t n, const T& value)
            : size_(n)
            , capacity_(n)
            , buf_(reinterpret_cast<T*>(new char[capacity_ * sizeof(T)])) {
        for (size_t i = 0; i < size_; ++i) {
            new (buf_ + i) T(value);
        }
    }

    explicit Vector(size_t n) : Vector(n, T()) {
    }

    Vector(std::initializer_list<T> list)
            : size_(list.size())
            , capacity_(size_)
            , buf_(reinterpret_cast<T*>(new char[capacity_ * sizeof(T)])) {
        std::copy(list.begin(), list.end(), buf_);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // operator=

    Vector<T>& operator=(const Vector& other) {
        if (&other != this) {
            Vector<T> copy(other);
            swap(copy);
        }
        return *this;
    }

    Vector<T>& operator=(Vector&& other) {
        if (&other != this) {
            Vector<T> copy(std::move(other));
            swap(copy);
        }
        return *this;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Capacity

    bool empty() const {
        return size_ == 0;
    }

    size_t size() const {
        return size_;
    }

    void reserve(size_t new_cap) {
        if (capacity_ >= new_cap) {
            return;
        }
        T* new_buf = nullptr;
        if (new_cap != 0) {
            new_buf = reinterpret_cast<T*>(new char[new_cap * sizeof(T)]);
        }

        for (size_t ind = 0; ind < size_; ++ind) {
            new (new_buf + ind) T(buf_[ind]);
            (buf_ + ind)->~T();
        }
        delete[] reinterpret_cast<char*>(buf_);

        buf_ = new_buf;
        capacity_ = new_cap;
    }

    size_t capacity() const {
        return capacity_;
    }

    void shrink_to_fit() {
        if (size_ == 0 && capacity_ > 0) {
            delete[] reinterpret_cast<char*>(buf_);
            buf_ = nullptr;
            capacity_ = 0;
        } else if (size_ != capacity_) {
            capacity_ = 0;
            reserve(size_);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Element access

    T& at(size_t pos) {
        if (pos >= size_) {
            throw std::out_of_range("Invalid index.");
        }
        return (*this)[pos];
    }

    const T& at(size_t pos) const {
        if (pos >= size_) {
            throw std::out_of_range("Invalid index.");
        }
        return (*this)[pos];
    }

    T& operator[](size_t pos) {
        return *(buf_ + pos);
    }

    const T& operator[](size_t pos) const {
        return *(buf_ + pos);
    }

    T& front() {
        return (*this)[0];
    }

    const T& front() const {
        return (*this)[0];
    }

    T& back() {
        return (*this)[size_ - 1];
    }

    const T& back() const {
        return (*this)[size_ - 1];
    }

    T* data() {
        return buf_;
    }

    const T* data() const {
        return buf_;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Modifiers

    void clear() {
        for (size_t ind = 0; ind < size_; ++ind) {
            (buf_ + ind)->~T();
        }
        size_ = 0;
    }

    void insert(size_t pos, const T& value) {
        if (pos > size_) {
            throw std::out_of_range("Invalid index.");
        }
        if (size_ == capacity_) {
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        }

        for (size_t ind = size_; ind > pos; --ind) {
            new (buf_ + ind) T(buf_[ind - 1]);
            (buf_ + ind - 1)->~T();
        }
        new (buf_ + pos) T(value);

        ++size_;
    }

    template<typename... Args>
    void emplace(size_t pos, Args&&... args) {
        if (pos > size_) {
            throw std::out_of_range("Invalid index.");
        }
        if (size_ == capacity_) {
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        }

        for (size_t ind = size_; ind > pos; --ind) {
            new (buf_ + ind) T(buf_[ind - 1]);
            (buf_ + ind - 1)->~T();
        }
        new (buf_ + pos) T(std::forward<Args>(args)...);

        ++size_;
    }

    void push_back(const T& value) {
        insert(size_, value);
    }

    template<typename... Args>
    void emplace_back(Args&&... args) {
        emplace(size_, std::forward<Args>(args)...);
    }

    void erase(size_t pos) {
        if (pos >= size_) {
            throw std::out_of_range("Invalid index.");
        }
        for (size_t ind = pos; ind < size_; ++ind) {
            (buf_ + ind)->~T();
            if (ind != size_ - 1) {
                new (buf_ + ind) T(buf_[ind + 1]);
            }
        }

        --size_;
    }

    void pop_back() {
        if (size_ == 0) {
            return;
        }
        erase(size_ - 1);
    }

    void resize(size_t count, const T& value = T()) {
        if (count < size_) {
            for (size_t ind = count; ind < size_; ++ind) {
                (buf_ + ind)->~T();
            }
        } else if (count > size_) {
            reserve(count);
            for (size_t ind = size_; ind < count; ++ind) {
                new (buf_ + ind) T(value);
            }
        }

        size_ = count;
    }

    void swap(Vector& other) {
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
        std::swap(buf_, other.buf_);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Destructor

    ~Vector() {
        clear();
        delete[] reinterpret_cast<char*>(buf_);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
private:
    size_t size_;
    size_t capacity_;
    T* buf_;
};