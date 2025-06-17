#ifndef MEDIAN_TREE_NODE
#define MEDIAN_TREE_NODE

#include "MedianTree.hpp"
#include "MedianTreeNodeData.hpp"

namespace MedianTree {

/* AVL Node class, holds data and references to adjacent nodes. */
template<typename T>
class Node {
public:
    friend class Tree<T>;

    Node(const T& newData, const double weight);
    ~Node();

private:
    // Shared pointers to the children
    std::shared_ptr<Node<T>> left;
    std::shared_ptr<Node<T>> right;

    // Weak pointer to the parent for traversal
    std::weak_ptr<Node<T>> parent;

    data_ptr<T> data;

    // Store these locally to reduce the time cost of lookups
    int left_height;
    int right_height;
    double left_weight;
    double right_weight;

    // If true this is the left child of its parent, False otherwise.
    bool left_child;

    int total_height();

    /**
     * @brief Update the internal parameters after a structure change.
     *
     * Only looks at the direct children. If they have changed, update must be
     * called on them first.
     **/
    void update();

    double total_weight();
};

template<typename T>
using node_ptr = std::shared_ptr<Node<T>>;

/* Constructor for Node, sets the node's data to element. */
template<typename T>
Node<T>::Node(const T& element, const double weight) :
    data(std::make_shared<Data<T>>(element, weight)) {
    // std::cout<<"new node: ("<<data<<", "<<weight<<")\n";
    left_height = 0;
    right_height = 0;
    left_weight = 0.0;
    right_weight = 0.0;
    left = nullptr;
    right = nullptr;
}

/* Destructor for Node. */
template<typename T>
Node<T>::~Node() {
    left = nullptr;
    right = nullptr;
}

/* Gets the total total_height of the subtree the node is the root of.
 * Adds together 1, left_height & right_height. */
template<typename T>
int Node<T>::total_height() {
    // return height + left_height + right_height;
    return std::max(left_height, right_height) + 1;
}

/* Gets the total total_weight of the subtree the node is the root of.
 * Adds together weight, left_weight & wight_height. */
template<typename T>
double Node<T>::total_weight() {
    // std::cout<<left_weight<<" + "<<right_weight<<" + "<<weight<<" = "<<left_weight + right_weight
    // + weight<<std::endl;
    return left_weight + right_weight + data->weight;
}

template<typename T>
void Node<T>::update() {
    left_height = left ? left->total_height() : 0;
    right_height = right ? right->total_height() : 0;

    left_weight = left ? left->total_weight() : 0.0;
    right_weight = right ? right->total_weight() : 0;
}

} // namespace MedianTree

#endif
