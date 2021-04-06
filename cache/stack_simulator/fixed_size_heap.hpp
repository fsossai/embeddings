#include <algorithm>
#include <cstdint>
#include <vector>
#include <unordered_map>

template<typename Tkey, typename Tval, typename Compare = std::greater<Tval>,
    typename Hasher = std::hash<Tkey>>
class FixedSizeHeap
{
public:
    FixedSizeHeap(uint64_t max_size);
    
    void heapify(uint64_t i);
    
    void insert(const Tkey& key, Tval val);
    
    template<typename Changer>
    void change(const Tkey& key, Changer changer);

    template<typename Changer>
    void change(Tkey&& key, Changer changer);

    uint64_t current_size();

    bool is_full();

    void swap(std::pair<Tkey, Tval>& a, std::pair<Tkey, Tval>& b);

    bool contains(const Tkey& key) const;

    bool contains(Tkey&& key) const;

    Tval& get(const Tkey& key);

    Tval& get(Tkey&& key);

    void set(const Tkey& key, const Tval& val);

    void set(const Tkey& key, Tval&& val);

private:
    std::vector<std::pair<Tkey, Tval>> _v;
    std::unordered_map<Tkey, uint64_t, Hasher> _m;
    Compare _comp;
    uint64_t _current_size = 0;
    uint64_t _max_size;

    inline uint64_t parent(uint64_t i) const;
    
    inline uint64_t left(uint64_t i) const;
    
    inline uint64_t right(uint64_t i) const;
};

/*** DEFINITIONS ***/

template<typename Tkey, typename Tval, typename Compare, typename Hasher>
FixedSizeHeap<Tkey, Tval, Compare, Hasher>::FixedSizeHeap(
    uint64_t max_size) :
    _comp(Compare()),
    _max_size(max_size),
    _v(max_size)
{ }

template<typename Tkey, typename Tval, typename Compare, typename Hasher>
inline uint64_t FixedSizeHeap<Tkey, Tval, Compare, Hasher>::parent(uint64_t i) const
{
    return (i - 1) / 2;
}

template<typename Tkey, typename Tval, typename Compare, typename Hasher>
inline uint64_t FixedSizeHeap<Tkey, Tval, Compare, Hasher>::left(uint64_t i) const
{
    return 2 * i + 1;
}

template<typename Tkey, typename Tval, typename Compare, typename Hasher>
inline uint64_t FixedSizeHeap<Tkey, Tval, Compare, Hasher>::right(uint64_t i) const
{
    return 2 * i + 2;
}
    
template<typename Tkey, typename Tval, typename Compare, typename Hasher>
void FixedSizeHeap<Tkey, Tval, Compare, Hasher>::heapify(uint64_t i)
{
    uint64_t l = left(i), r = right(i), pivot;

    if (l < _current_size)
        pivot = (_comp(_v[i].second, _v[l].second) ? i : l);
    else
        pivot = i;
    if (r < _current_size)
        pivot = (_comp(_v[pivot].second, _v[r].second) ? pivot : r);
    if (pivot != i)
    {
        _m[_v[pivot].first] = i;
        _m[_v[i].first] = pivot;
        swap(_v[pivot], _v[i]);
        heapify(pivot);
    }
}

template<typename Tkey, typename Tval, typename Compare, typename Hasher>
void FixedSizeHeap<Tkey, Tval, Compare, Hasher>::insert(
    const Tkey& key, Tval val)
{
    if (_current_size == _max_size)
    {
        _m.erase(_v[0].first);
        _v[0] = _v[--_current_size];
        heapify(0);
    }
    _v[_current_size] = std::make_pair(key, val);
    _m.insert(std::make_pair(key, _current_size));
    uint64_t i = _current_size;
    ++_current_size;
  
    // positioning a node in the heap by swapping elements
    while (i && !_comp(_v[parent(i)].second, _v[i].second))
    {
        _m[_v[i].first] = parent(i);
        _m[_v[parent(i)].first] = i;
        swap(_v[i], _v[parent(i)]);
        i = parent(i);
    }
}

template<typename Tkey, typename Tval, typename Compare, typename Hasher>
template<typename Changer>
void FixedSizeHeap<Tkey, Tval, Compare, Hasher>::change(
    const Tkey& key,
    Changer changer)
{
    //assert(contains(key));
    uint64_t i = _m[key];
    _v[i].second = changer(_v[i].second);
    heapify(i);
}

template<typename Tkey, typename Tval, typename Compare, typename Hasher>
template<typename Changer>
void FixedSizeHeap<Tkey, Tval, Compare, Hasher>::change(
    Tkey&& key,
    Changer changer)
{
    //assert(contains(key));
    uint64_t i = _m[key];
    _v[i].second = changer(_v[i].second);
    heapify(i);
}

template<typename Tkey, typename Tval, typename Compare, typename Hasher>
uint64_t FixedSizeHeap<Tkey, Tval, Compare, Hasher>::current_size()
{
    return _current_size;
}


template<typename Tkey, typename Tval, typename Compare, typename Hasher>
bool FixedSizeHeap<Tkey, Tval, Compare, Hasher>::is_full()
{
    return _current_size == _max_size;
}

template<typename Tkey, typename Tval, typename Compare, typename Hasher>
void FixedSizeHeap<Tkey, Tval, Compare, Hasher>::swap(
    std::pair<Tkey, Tval>& a,
    std::pair<Tkey, Tval>& b)
{
    std::pair<Tkey, Tval> temp = a;
    a = b;
    b = temp;
}

template<typename Tkey, typename Tval, typename Compare, typename Hasher>
bool FixedSizeHeap<Tkey, Tval, Compare, Hasher>::contains(const Tkey& key) const
{
    return (_m.find(key) != _m.end());
}

template<typename Tkey, typename Tval, typename Compare, typename Hasher>
bool FixedSizeHeap<Tkey, Tval, Compare, Hasher>::contains(Tkey&& key) const
{
    return (_m.find(key) != _m.end());
}

template<typename Tkey, typename Tval, typename Compare, typename Hasher>
Tval& FixedSizeHeap<Tkey, Tval, Compare, Hasher>::get(const Tkey& key)
{
    return _v[_m[key]].second;
}

template<typename Tkey, typename Tval, typename Compare, typename Hasher>
Tval& FixedSizeHeap<Tkey, Tval, Compare, Hasher>::get(Tkey&& key)
{
    return _v[_m[key]].second;
}

template<typename Tkey, typename Tval, typename Compare, typename Hasher>
void FixedSizeHeap<Tkey, Tval, Compare, Hasher>::set(const Tkey& key, const Tval& val)
{
    change(
        key,
        [&val](const Tval& old)
        {
            return val;
        }
    );
}

template<typename Tkey, typename Tval, typename Compare, typename Hasher>
void FixedSizeHeap<Tkey, Tval, Compare, Hasher>::set(const Tkey& key, Tval&& val)
{
    change(
        key,
        [val](const Tval& old)
        {
            return val;
        }
    );
}

