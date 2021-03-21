#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>

#include "rank_tree.hpp"

template<typename T>
RankTreeNode<T>::RankTreeNode(T name) {
	_name = name;
	_weight = 0;
	_priority = rand();
	_left = nullptr;
	_right = nullptr;
	_parent = nullptr;
}

template<typename T>
RankTreeNode<T>::~RankTreeNode() {
	if (_left != nullptr) {
		delete _left;
	}
	if (_right != nullptr) {
		delete _right;
	}
}

template<typename T>
void RankTreeNode<T>::print(int level) {
	for (int i=0; i< level; i++) {
		cout << "  ";
	}
	cout << _name 
	<< " " << _weight 
	<< " " << _priority 
	<< " " << Rank()
	<< endl;
	if (_left) {
		_left->print(level+1);
	} else { 
		for (int i=0; i< level+1; i++) {
			cout << "  ";
		}
		cout << "<empty subtree>\n";
	}

	if (_right) {
		_right->print(level+1);
	} else { 
		for (int i=0; i< level+1; i++) {
			cout << "  ";
		}
		cout << "<empty subtree>\n";
	}

}

template<typename T>
uint64_t RankTreeNode<T>::checkWeights() {
	uint64_t leftWeight = 0;
	uint64_t rightWeight = 0;
	if (_left != nullptr) {
		leftWeight = _left->checkWeights();
	}
	if (_right != nullptr) {
		rightWeight = _right->checkWeights();
	}
	assert(_weight == 1 + leftWeight + rightWeight);
	return _weight;
}

template<typename T>
void RankTreeNode<T>::checkPriorities(int priority) {
	assert(_priority <= priority);
	if (_left != nullptr) {
		_left->checkPriorities(_priority);
	}
	if (_right != nullptr) {
		_right->checkPriorities(_priority);
	}
}


template<typename T>
void RankTreeNode<T>::checkParentPointers(RankTreeNode* parent) {
	assert(_parent == parent);
	if (_left != nullptr) {
		_left->checkParentPointers(this);
	}
	if (_right != nullptr) {
		_right->checkParentPointers(this);
	}
}


template<typename T>
void RankTreeNode<T>::checkUniqueness(std::set<RankTreeNode<T>*>& ptrs) {
	assert(ptrs.count(this) == 0);
	auto start_weight = ptrs.size();
	ptrs.insert(this);
	if (_left != nullptr) {
		_left->checkUniqueness(ptrs);
	}
	if (_right != nullptr) {
		_right->checkUniqueness(ptrs);
	}
	assert(ptrs.size() == start_weight + _weight);
}

template<typename T>
void RankTreeNode<T>::promote() {
	assert(_parent != nullptr);
	auto A = _parent;
	auto GP = _parent->_parent;
	if (leftChild()) {
		auto E = _right;
		_right = A;
		A->_parent = this;
		A->_left = E;
		if (E != nullptr) {
			E->_parent = A;
		}
	} else {
		assert(rightChild());
		auto E = _left;
		_left = A;
		A->_parent = this;
		A->_right = E;
		if (E != nullptr) {
			E->_parent = A;
		}
	}
	_parent = GP;
	if (GP != nullptr) {
		if (GP->_left == A) {
			GP->_left = this;
		} else {
			assert(GP->_right == A);
			GP->_right = this;
		}
	} 
}

template<typename T>
RankTreeNode<T>* RankTreeNode<T>::demote() {
	assert(!leaf());
	RankTreeNode* ret;
	if (_left == nullptr) {
		ret = _right;
	} else if (_right == nullptr) {
		ret = _left;
	} else {
		assert(_left != nullptr && _right != nullptr);
		if (_left->_priority >= _right->_priority) {
			ret = _left;
		} else {
			ret = _right;
		}
	}
	ret->promote();
	return ret;
}


template<typename T>
void RankTreeNode<T>::unlink() {
	assert(leaf());
	if (_parent == nullptr) {
		return;
	}
	if (_parent->_right == this) {
		_parent->_right = nullptr;
	} else {
		assert(_parent->_left == this);
		_parent->_left = nullptr;
	}
	_parent->fixWeights();
}


template<typename T>
RankTreeNode<T>* RankTreeNode<T>::makeLeaf() {
	RankTreeNode<T>* ret = nullptr;
	while(!leaf()) {
		if (root()) {
			ret = demote();
		} else {
			demote();
		}
	}
	return ret;
}

template<typename T>
bool RankTreeNode<T>::fixPriority() {
	while(_parent != nullptr && _parent->_priority < _priority) {
		promote();
	}
	if (_parent == nullptr) {
		return true;
	} else {
		return false;
	}
}

template<typename T>
uint64_t RankTreeNode<T>::leftChildRank(uint64_t rank) {
	if (root()) {
		return rank;
	} else if (leftChild()) {
		return _parent->leftChildRank(rank);
	} else {
		assert(rightChild());
		return _parent->rightChildRank(rank);
	}
}

template<typename T>
uint64_t RankTreeNode<T>::rightChildRank(uint64_t rank) {
	if (root()) {
		return 1 + rank + leftWeight();
	} else if (leftChild()) {
		return _parent->leftChildRank(1 + rank + leftWeight());
	} else {
		assert(rightChild());
		return _parent->rightChildRank(1 + rank + leftWeight());
	}
}

template<typename T>
uint64_t RankTreeNode<T>::Rank() {
	if (root()) {
		return leftWeight();
	} else if (leftChild()) {
		return _parent->leftChildRank(leftWeight());
	} else {
		assert(rightChild());
		return _parent->rightChildRank(leftWeight());
	}
}

template<typename T>
RankTree<T>::RankTree() {
	_root = nullptr;
}

template<typename T>
RankTree<T>::~RankTree() {
	if (_root != nullptr) {
		delete _root;
	}
}

template<typename T>
RankTreeNode<T>* RankTree<T>::first() {
	if (_root == nullptr) {
		return nullptr;
	}
	RankTreeNode<T>* node = _root;
	while (node->_left != nullptr) {
		node = node->_left;
	}
	return node;
}

template<typename T>
RankTreeNode<T>* RankTree<T>::last() {
	if (_root == nullptr) {
		return nullptr;
	}
	RankTreeNode<T>* node = _root;
	while(node->_right != nullptr) {
		node = node->_right;
	}
	return node;
}


template<typename T>
bool RankTreeNode<T>::leaf() {
	return _left == nullptr && _right == nullptr;
}


template<typename T>
bool RankTreeNode<T>::root() {
	return _parent == nullptr;
}


template<typename T>
bool RankTreeNode<T>::leftChild() {
	return (_parent != nullptr) && (_parent->_left == this);
}


template<typename T>
bool RankTreeNode<T>::rightChild() {
	return (_parent != nullptr) && (_parent->_right == this);
}


template<typename T>
uint64_t RankTreeNode<T>::leftWeight() {
	if (_left != nullptr) {
		return _left->_weight;
	} else {
		return 0;
	}
}


template<typename T>
uint64_t RankTreeNode<T>::rightWeight() {
	if (_right != nullptr) {
		return _right->_weight;
	} else {
		return 0;
	}
}


template<typename T>
void RankTreeNode<T>::fixWeights() {
	_weight = 1 + leftWeight() + rightWeight();
	if (_parent != nullptr) {
		_parent->fixWeights();
	}
}


template<typename T>
RankTreeNode<T>* RankTree<T>::Insert(T name) {
	auto node = new RankTreeNode(name);
	InsertNode(node);
	return node;
}


template<typename T>
void RankTree<T>::InsertNode(RankTreeNode<T>* node) {
	if (_root == nullptr) {
		_root = node;
	} else {
		auto first_ = first();
		first_->_left = node;
		node->_parent = first_;
		if(node->fixPriority()) {
			_root = node;
		}
	}
	node->fixWeights();
}


template<typename T>
void RankTree<T>::print() {
	if (_root == nullptr) {
		cout << "<Empty tree>\n";
		return;
	}
	_root->print(0);
}


template<typename T>
void RankTree<T>::check() {
	if (_root != nullptr) {
		_root->checkPriorities(_root->_priority);
		_root->checkParentPointers(nullptr);
		_root->checkWeights();
		std::set<RankTreeNode<T>*> nodes;
		_root->checkUniqueness(nodes);
	}
}


template<typename T>
void RankTree<T>::Remove(RankTreeNode<T>* node) {
	auto newRoot = node->makeLeaf();
	if (newRoot != nullptr) {
		_root = newRoot;
	}
	if (_root == node) {
		_root = nullptr;
	} else {
		node->unlink();
	}
}


template<typename T>
uint64_t RankTree<T>::computeSize() {
	if (_root == nullptr) {
		return 0;
	} else {
		return _root->checkWeights();
	}
}

#include "rank_tree.spec.cpp"