#ifndef MEDIAN_NODE_DATA
#define MEDIAN_NODE_DATA

#include "MedianTree.hpp"

namespace MedianTree {

// Forward declarations
template<typename T>
class Tree;

template<typename T>
class Node;

template<typename T>
class Data {
public:
    friend class Tree<T>;
    friend class Node<T>;

    Data(const T& value, const double weight);
    double weight;

private:
    T value;

    inline bool operator<(const Data<T> c) const;
    inline bool operator==(const Data<T> c) const;
};

template<typename T>
using data_ptr = std::shared_ptr<Data<T>>;

template<typename T>
Data<T>::Data(const T& value, const double weight) {
    this->value = value;
    this->weight = weight;
}

template<typename T>
inline bool Data<T>::operator<(const Data<T> c) const {
    if (value == c.value)
        return weight < c.weight;
    return value < c.value;
}

template<typename T>
inline bool Data<T>::operator==(const Data<T> c) const {
    if (value == c.value && weight == c.weight)
        return true;
    return false;
}

} // namespace MedianTree

#endif
