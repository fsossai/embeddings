# Simulating Embedding Lookups in a Distributed System with Caches

The system is composed of _P_ distributed machines, each one equipped with a **processing unit**,
and a **main memory** and additional space for a **cache** memory.
Every memory stores a partition of the set of embedding tables necessary for a Deep Learning model
such as DLRM.

We assume that the main memory corresponds to the DRAM of the processing unit and the cache
is part of it, so that they form a **Uniform Memory Access** (UMA) region.

The simulation is written in C++17.

