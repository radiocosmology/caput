/*
 * The MIT License (MIT)
 * Copyright (c) 2016 Ethan Gaebel
 *               2019 Richard Shaw
 *               2019 Rick Nitsche
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 * and associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef MEDIAN_TREE
#define MEDIAN_TREE

#include "MedianTreeNode.hpp"
#include "MedianTreeNodeData.hpp"

#include <cmath>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace MedianTree {

// Forward declarations
template<typename T>
class Tree;

template<typename T>
class Node;

// The maximum imbalance that is allowed in the Tree.
const static int MAX_IMBAL = 1;

template<typename T>
class Tree {
public:
    /**
     * @brief Constructor
     */
    Tree();

    /**
     * @brief insert Insert an element with a weight into the tree.
     *
     * @param element Element to insert.
     * @param weight Weight to insert with the element.
     * @return A pointer to the element in the tree.
     */
    inline data_ptr<T> insert(const T& element, const double weight);

    /**
     * @brief remove Remove an element from the tree.
     *
     * @param element The element to remove.
     *
     * @return False in case the element was not found.
     */
    bool remove(const data_ptr<T> element);

    /**
     * @brief clear Remove all elements from the tree.
     */
    void clear();

    /**
     * @brief size Get the tree's size.
     *
     * @return Size of the tree.
     */
    int size();

    /**
     * @brief Get weighted median of tree values.
     *
     * @param method If the conditions for finding the weighted median is true for more than one,
     * this decides how the median is found:
     * 's': the average of all candidates is used
     * 'l': the lowest value is used
     * 'u': the highest value is used
     *
     * @return The weighted median of the whole tree.
     */
    double weighted_median(char method);


    /*************************************************
     * Functions for testing and debugging the tree. *
     *************************************************/

    /**
     * @brief Check the tree is balanced.
     *
     * @return  True if the tree is balanced.
     **/
    bool check_balance();

    /**
     * @brief Print balance properties of the tree.
     **/
    void print_balance();

    /**
     * @brief Print the tree as an ordered list.
     **/
    void print_list();

    /**
     * @brief Print the tree.
     **/
    void print_tree();

    /**
     * @brief Get the items in a vector.
     **/
    std::vector<T> get_vector();


private:
    double weighted_median_split(node_ptr<T> node, double previous_weights, double next_weights,
                                 double midpoint);
    double weighted_median_lower(node_ptr<T> node, double previous_weights, double next_weights,
                                 double midpoint);
    double weighted_median_higher(node_ptr<T> node, double previous_weights, double next_weights,
                                  double midpoint);
    node_ptr<T> find_max();
    node_ptr<T> find_min();
    node_ptr<T> in_order_successor(node_ptr<T> node);
    node_ptr<T> in_order_predecessor(node_ptr<T> node);
    node_ptr<T> find_min(node_ptr<T> node);
    node_ptr<T> find_max(node_ptr<T> node);
    const data_ptr<T>* find(const data_ptr<T> element);

    node_ptr<T> root;
    int tree_size;

    /**
     * @brief Get a reference to the pointer that references the given node.
     *
     * @param  node  Node we are interested in.
     * @return       The pointer that refers to me, either a pointer on the
     *               parent or the root pointer.
     **/
    node_ptr<T>& pointer_to(const Node<T>& node);

    /**
     * @brief Remove a given node.
     *
     * This uses the usual method to remove a node from a BST but does not
     * perform any rebalancing.
     *
     * @param  node  The node to remove.
     * @return       The lowest modified node. This is the point from which we
     *               need to rebalance.
     **/
    node_ptr<T> remove_node(Node<T>& node);

    node_ptr<T> find_node(node_ptr<T> node, const data_ptr<T> element);
    void balance(node_ptr<T>& node);
    void balance_path(node_ptr<T>& node);
    void rotate_L(node_ptr<T>& node);
    void rotate_R(node_ptr<T>& node);
    node_ptr<T> find(const node_ptr<T>& node, const data_ptr<T>& element);
    using traverse_func = std::function<void(Node<T>&, unsigned int)>;
    enum class traverse_order { IN, OUT, POST };

    /**
     * @brief Traverse the tree applying a function to each node.
     *
     * @param  node   Node to start from.
     * @param  f      Function to apply.
     * @param  order  Which order to traverse the array.
     * @param  depth  What is the depth of this node from the root.
     **/
    void traverse(const node_ptr<T>& node, traverse_func f,
                  traverse_order order = traverse_order::IN, unsigned int depth = 0);
};

/* AVLTree constructor, sets root to nullptr. */
template<typename T>
Tree<T>::Tree() : root(nullptr), tree_size(0) {}

/* Helper method for inserting the passed element into the AVLTree. */
template<typename T>
inline data_ptr<T> Tree<T>::insert(const T& element, const double weight) {
    if (root == nullptr) {
        root = std::make_shared<Node<T>>(element, weight);
        return root->data;
    }

    node_ptr<T> node = root;
    node_ptr<T> new_node = std::make_shared<Node<T>>(element, weight);

    while (true) {
        if (*new_node->data < *node->data) {
            if (node->left)
                node = node->left;
            else {
                node->left = new_node;
                node->left->parent = node;
                new_node->left_child = true;
                break;
            }
        } else {
            if (node->right)
                node = node->right;
            else {
                node->right = new_node;
                node->right->parent = node;
                new_node->left_child = false;
                break;
            }
        }
    }

    balance_path(node);
    tree_size++;
    return new_node->data;
}

/* Searches the AVLTree for the passed in element.
 * Returns nullptr if the element cannot be found.
 */
template<typename T>
const data_ptr<T>* Tree<T>::find(const data_ptr<T> element) {
    auto node = find_node(root, element);
    return node ? &(node->data) : nullptr;
}

/* Helper method for finding the passed in element in the subTree with sub_root
 * as the root. */
template<typename T>
node_ptr<T> Tree<T>::find_node(node_ptr<T> node, const data_ptr<T> element) {
    while (node != nullptr) {
        if (*node->data == *element)
            return node;
        if (*element < *node->data)
            node = node->left;
        else
            node = node->right;
    }
    return nullptr;
}

/* Helper method to remove the passed element from the AVLTree (if it is
 * present). */
template<typename T>
bool Tree<T>::remove(const data_ptr<T> element) {
    // Find the node to remove.
    auto target_node = find_node(root, element);

    // If the element is not found, return False
    if (!target_node)
        return false;

    node_ptr<T> successor = remove_node(*target_node);

    balance_path(successor);
    tree_size--;
    return true;
}


template<typename T>
node_ptr<T> Tree<T>::remove_node(Node<T>& node) {
    // No children of the found node, remove it.
    if (!node.left && !node.right) {
        node_ptr<T> parent = node.parent.lock();
        if (node.left_child)
            parent->left = nullptr;
        else
            parent->right = nullptr;
        return parent;
    }

    // Node has one child. Replace the node with the child
    if (!node.left || !node.right) {
        node_ptr<T> child = node.left ? node.left : node.right;
        pointer_to(node) = child;
        node_ptr<T> parent = node.parent.lock();
        child->parent = parent;
        child->left_child = node.left_child;
        return parent;
    }

    // Node has two children. Find the successor, swap with the node to delete,
    // and delete the original place of the successor
    node_ptr<T> successor = find_min(node.right);
    node.data = successor->data;
    return remove_node(*successor);
}

template<typename T>
node_ptr<T>& Tree<T>::pointer_to(const Node<T>& node) {
    node_ptr<T> parent = node.parent.lock();
    if (!parent)
        return root;

    if (node.left_child)
        return parent->left;
    return parent->right;
}

/* Check the balance of the AVLTree that has node as its root. */
template<typename T>
void Tree<T>::balance(node_ptr<T>& node) {
    switch (node->left_height - node->right_height) {
        case MAX_IMBAL + 1:
            // Single rotation is necessary if the left-subtree is larger than the
            // right.
            if (node->left->left_height >= node->left->right_height) {
                rotate_R(node);
                return;
            }
            // Double rotation is necessary if the right-subtree is larger than the
            // left.
            rotate_L(node->left);
            rotate_R(node);
            return;
        case -(MAX_IMBAL + 1):
            // If the right subTree is too large.
            // Single rotation is necessary if the right-subtree is larger than the
            // left.
            if (node->right->right_height >= node->right->left_height) {
                rotate_L(node);
                return;
            }
            // Double rotation is necessary if the left sub-tree is larger than the
            // right.
            rotate_R(node->right);
            rotate_L(node);
        default:
            return;
    }
}


template<typename T>
void Tree<T>::balance_path(node_ptr<T>& node) {
    while (node) {
        node->update();
        balance(pointer_to(*node));
        node = node->parent.lock();
    }
}

/* Rotate the left subTree over to the root. */
template<typename T>
void Tree<T>::rotate_R(node_ptr<T>& node) {
    node_ptr<T> temp = node->left;

    node->left = temp->right;
    if (temp->right) {
        temp->right->parent = node;
        temp->right->left_child = true;
    }

    temp->right = node;
    temp->left_child = node->left_child;
    node->left_child = false;
    temp->parent = node->parent;
    node->parent = temp;

    node->update();
    temp->update();

    node = temp;
}

/* Rotate the right subTree over to the root. */
template<typename T>
void Tree<T>::rotate_L(node_ptr<T>& node) {
    node_ptr<T> temp = node->right;

    node->right = temp->left;

    if (temp->left) {
        temp->left->parent = node;
        temp->left->left_child = false;
    }

    temp->left = node;
    temp->left_child = node->left_child;
    node->left_child = true;
    temp->parent = node->parent;
    node->parent = temp;

    node->update();
    temp->update();


    node = temp;
}

/* Finds the node with the minimum element in the AVLTree. */
template<typename T>
node_ptr<T> Tree<T>::find_min() {
    return root ? pointer_to(root.find_min()) : nullptr;
}

/* Finds node with the the maximum element in the AVLTree. */
template<typename T>
node_ptr<T> Tree<T>::find_max() {
    return root ? pointer_to(root.find_max()) : nullptr;
}

/* Helper function for inOrderTraversal. Performs an inOrderTraversal of the
 * subTree with sub_root as its root. */
template<typename T>
void Tree<T>::print_tree() {

    auto pn = [&](Node<T>& node, unsigned int depth) {
        for (unsigned int i = 0; i < depth; i++)
            std::cout << "----";
        std::cout << "|"
                  << "(" << node.data->value << ", " << node.data->weight << ")" << std::endl;
    };

    traverse(root, pn, traverse_order::OUT);
    std::cout << std::endl;
}

/* Gets the size of the AVLTree. */
template<typename T>
int Tree<T>::size() {
    return tree_size;
}

/* Empties out the AVLTree. */
template<typename T>
void Tree<T>::clear() {
    delete_traversal(root);
    root = nullptr;
    tree_size = 0;
}

template<typename T>
void Tree<T>::traverse(const node_ptr<T>& node, traverse_func f, traverse_order order,
                       unsigned int depth) {
    if (!node)
        return;

    switch (order) {
        case traverse_order::IN:
            traverse(node->left, f, order, depth + 1);
            f(*node, depth);
            traverse(node->right, f, order, depth + 1);
            break;
        case traverse_order::OUT:
            traverse(node->right, f, order, depth + 1);
            f(*node, depth);
            traverse(node->left, f, order, depth + 1);
            break;
        case traverse_order::POST:
            traverse(node->left, f, order, depth + 1);
            traverse(node->right, f, order, depth + 1);
            f(*node, depth);
            break;
    }
}

/* Check the tree for balance */
template<typename T>
bool Tree<T>::check_balance() {
    bool balanced = true;
    auto cb = [&](Node<T>& node, unsigned int depth) {
        bool node_balanced = (std::abs(node.left_height - node.right_height) <= MAX_IMBAL);
        balanced = (balanced && node_balanced);
    };
    traverse(root, cb, traverse_order::POST);
    if (!balanced)
        throw std::runtime_error("Not balanced");
    return balanced;
}

/* Check the tree for balance */
template<typename T>
void Tree<T>::print_balance() {
    auto pb = [](Node<T>& node, unsigned int depth) {
        node.update();
        std::cout << std::setw(5) << node->data.value << "  " << std::setw(6) << node.total_height()
                  << "  " << std::setw(7) << node.left_height - node.right_height << std::endl;
    };

    std::cout << "Value  Height  Balance" << std::endl << "----------------------" << std::endl;
    traverse(root, pb, traverse_order::OUT);
}


/* Print the list of items */
template<typename T>
void Tree<T>::print_list() {
    Tree<T>::traverse_func pf = [](Node<T>& node, unsigned int depth) {
        std::cout << "(" << node.data->value << ", " << node.data->weight << "), ";
    };

    std::cout << "[";
    traverse(root, pf, traverse_order::IN);
    std::cout << "]\n";
}

/* Get the weighted median                                                   *
 *                                                                           *
 * The weighted median is defined as the value i in an ordered sequence,     *
 * where the sum of the weights of the values 0..i-1 is smaller or equal half the *
 * sum of all weights 0..n and the sum of the sum of the weights i+1..n is   *
 * smaller or equal half the sum of all weights 0..n. The weights are assumed to  *
 * be order according to the values.                                         */
template<typename T>
double Tree<T>::weighted_median(char method) {
    if (root == nullptr)
        return 0;
    node_ptr<T> node = root;
    double midpoint = root->total_weight() / 2.0;

    // cumulated weights of all next & previous nodes (this can include parents and siblings)
    double next_weights = 0;
    double previous_weights = 0;

    // Find the highest order node that satisfies median conditions
    while (true) {
        if (node->left_weight + previous_weights > midpoint) {
            next_weights += node->right_weight + node->data->weight;
            node = node->left;
        } else if (node->right_weight + next_weights > midpoint) {
            previous_weights += node->left_weight + node->data->weight;
            node = node->right;
        } else
            break;
    }

    if (method == 's')
        return weighted_median_split(node, previous_weights, next_weights, midpoint);
    if (method == 'l')
        return weighted_median_lower(node, previous_weights, next_weights, midpoint);
    if (method == 'h')
        return weighted_median_higher(node, previous_weights, next_weights, midpoint);
    throw std::invalid_argument("Method must be 's', 'l' or 'h', but was " + method);
}

template<typename T>
double Tree<T>::weighted_median_split(node_ptr<T> node, double previous_weights,
                                      double next_weights, double midpoint) {
    node_ptr<T> top_median = node;
    double median = (double)node->data->value;
    size_t splt_median_cnt = 1;

    while (node->right_weight + next_weights + node->data->weight <= midpoint) {
        next_weights += node->data->weight + node->right_weight;
        node = in_order_predecessor(node);
        if (!node)
            break;
        splt_median_cnt++;
        median += node->data->value;
    }
    node = top_median;
    while (node->left_weight + previous_weights + node->data->weight <= midpoint) {
        previous_weights += node->data->weight + node->left_weight;
        node = in_order_successor(node);
        if (!node)
            break;
        splt_median_cnt++;
        median += node->data->value;
    }
    return median / splt_median_cnt;
}

template<typename T>
double Tree<T>::weighted_median_higher(node_ptr<T> node, double previous_weights,
                                       double next_weights, double midpoint) {
    // Take the highest value that qualifies.
    while (node->right_weight + next_weights <= midpoint
           && node->left_weight + previous_weights + node->data->weight <= midpoint && node->right)
        node = node->right;
    return (double)node->data->value;
}

template<typename T>
double Tree<T>::weighted_median_lower(node_ptr<T> node, double previous_weights,
                                      double next_weights, double midpoint) {
    // Take the lowest value that qualifies.
    while (node->left_weight + previous_weights <= midpoint
           && node->right_weight + next_weights + node->data->weight <= midpoint && node->left)
        node = node->left;
    return (double)node->data->value;
}

/* Get the vector of items */
template<typename T>
std::vector<T> Tree<T>::get_vector() {

    std::vector<T> list;

    Tree<T>::traverse_func vf = [&](Node<T>& node, unsigned int depth) {
        list.push_back(node.data->value);
    };

    traverse(root, vf, traverse_order::IN);

    return list;
}

template<typename T>
node_ptr<T> Tree<T>::in_order_successor(node_ptr<T> node) {
    if (node->right)
        return find_min(node->right);

    node_ptr<T> p = node->parent.lock();
    while (p && node == p->right) {
        node = p;
        p = p->parent.lock();
    }
    return p;
}

template<typename T>
node_ptr<T> Tree<T>::in_order_predecessor(node_ptr<T> node) {
    if (node->left)
        return find_max(node->left);

    node_ptr<T> p = node->parent.lock();
    while (p && node == p->left) {
        node = p;
        p = p->parent.lock();
    }
    return p;
}

template<typename T>
node_ptr<T> Tree<T>::find_min(node_ptr<T> node) {
    while (node->left) {
        node = node->left;
    }
    return node;
}

template<typename T>
node_ptr<T> Tree<T>::find_max(node_ptr<T> node) {
    while (node->right) {
        node = node->right;
    }
    return node;
}

} // namespace MedianTree
#endif
