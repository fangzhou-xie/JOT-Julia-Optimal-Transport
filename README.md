# JOT Julia Optimal Transport

<!-- This is a JOT (Julia Optimal Transport) -->
Julia is a new and promising language and
I plan to cover some algorithms in Optimal Transport
with implementation of Julia.
To the best of my knowledge, there does not exist such package in Julia
communities.

Given Julia's performance, it should run faster than other
high-level languages (for example Python).
Whereas OT problems often suffers from its computational complexity
in high-dimensional space. Therefore I consider Julia to be
a good choice of solving OT problems.

The first algorithm is Sinkhorn-Knopp (SK) matrix-scaling algorithm.
It will be translated from POT (Python Optimal Transport) library,
and I will compare running time given two different languages.
SK algorithm in POT is written using *numpy* library,
but Julia are supposed to be even faster.
<!-- and I wonder if Julia implementation is faster. -->
<!-- which is based on C/C++ and Fortran as its base. -->


TO DO:  
Recent paper for benchmark test of OT algorithms.   
Schrieber, J., Schuhmacher, D., & Gottschlich, C. (2017). DOTmark -
A Benchmark for Discrete Optimal Transport. IEEE Access, 5, 271–282.
https://doi.org/10.1109/ACCESS.2016.2639065
