# Discover

User: DAU, ingestion/query frequency, session duration

Functional Requirements: UX, feature, session memory, continous improvement

Data

* Numberic: min/max, negative, zero, null, `float("inf")`, if 2+ params then magnitude difference
* Array: emtpy, null, single element, all same, (un)sorted, (non)decreasing, (non)constant change, outliers at end
* String: ANSII, special chars
 
System: CPU time / memory / I/O operation constraints

Quality: test first?

Trade-off: speed vs memory vs accuracy, batch vs real-time (stream?), CAP


# Design

[BigO](https://www.bigocheatsheet.com/)

Brute force (verbal only)

* enumerate/simulate then sort & filter
* identify bottleneck: duplicate/expensive calculation, unnecessary/temporary storage

Optimised (verbal then code)

* Python idiomatic: list comprehension, generative `yield` function, swap operator (`a,b=b,a`)


# Develop

class @staticmethod

edge-case first

side effect idempotency (mutable input objects)

helper functions, Single Responsibility Principle (SRP) 

meaningful white space, comments


# Data Stuctures

inmutable: int, float, str, tuple

mutable: set, list, hashmap

operations: 
* create, destroy
* read at index/start/end (peak)
* insert/update at index/end/start
* delete at index/end/start
* find by key
* sort by key, reverse
* size, is_empty

## Hashmap

hash_function(key) = value, key must be inmutable

expected insertion O(1), but if collision (hash_fun(key1) == hash_fun(key2)), resolution: value is collection of n items, insertion O(n)

read is O(1)

```python
map = dict()

from collections import defaultdict

map_n = defaultdict(int)
asset map_n[key] == 0

map_l = defaultdict(list)
asset map_l[key] == []

from collections import OrderedDict

sorted_map = OrderedDict()
sorted_map['a'] = 1
sorted_map['b'] = 2
[assert k == "ab"[i] for i, k in enumerate(sorted_map.keys()]

from collections import Counter
counter_map = Counter(['abbccc')
assert counter_map['a'] == 1
assert counter_map['b'] == 2
assert counter_map['c'] == 3
```

## Set

## Array

```python
arr = []
assert typeof(arr) == list
arr.append(val)
arr.pop()  # pop_right
arr.extend([val_1, val_2])
```

## Linked List

### Singly Linked List

```python
class Node:
    val: int
    next: "Node"

    def __init__(self, val: int, next: Node = None):
        self.val = val
        self.next = next

head = Node(1, None)
next = Node(2, None)
head.next = next

tail = Node(3, None)
next.next = tail
```

### Doubly Linked List

## Stack (LIFO)

## Queue (FIFO)

## Tree

```python
class TrieNode:
    val: int
    children: list[Node]

    def __init__(self, val: int, children: list[Node] = None):
        self.val = val
        self.children = children
```

* balanced
* binary tree: node = (val, left, right)
* binary search: left.val <= curr.val < right.val,
* perfect binary
* red-black
* AVL
* complete binary
* full binary
* trie

## Heap

```python
from heapq import heapify, heappush, heappop
min_heap = []
[heappush(heap, i) for i in [1,2,3]]
min_heap = heapify([1, 2, 3])
min_val = heappop(min_heap)

min_heap = heapify([(1, 'a'), (2, 'b')])  # (val: num,label: any)
heappush(min_heap, (3, 'c'))
while min_heap:
    (val, label) = heappop(min_heap)

max_heap = []
[heappush(heap, -1 * i) for i in [1,2,3]]
max_heap = heapify([-1, -2, -3])
max_val = -1 * heappop(max_heap)
```

binary heap

## Algorithms

### DFS

LIFO, O(n) or O(h) where h is depth

```python
def dft(head: Node) -> set:
    seen, stack = set(), [head]
    while stack:
        curr = stack.pop()
        if curr not in seen:
            seen.add(curr)
            stack.extend([n for n in curr.neighbours])
    return seen
```

### BFS

```python
from collections import dequeue
def bfs(head: Node) -> set:
    seen, queue = {head}, dequeue([head])
    while queue:
        curr = queue.pop()
        for n in curr.neighbours:
            seen.add(n)
            queue.append(n)
    return seen
```

### Dijkstra's algorithm

topological sort

### Kruskal's algorithm

### Prim's algorithms

### Tarjan's algorithm

### Floyd-Warshall algorithm
