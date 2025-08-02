# distutils: language = c++
# cython: language_level = 2
from libcpp cimport bool
from libcpp.memory cimport shared_ptr

cdef extern from "MedianTree.hpp" namespace "MedianTree":
    cdef cppclass Tree[T]:
        Tree() nogil
        shared_ptr[Data[T]] insert(const T& element, const double weight) nogil
        bool remove(const shared_ptr[Data[T]] node) nogil
        T weighted_median(char method) nogil
        int size()

cdef extern from "MedianTreeNodeData.hpp" namespace "MedianTree":
    cdef cppclass Data[T]:
        Data(const T& value, const double weight) nogil
        double weight
