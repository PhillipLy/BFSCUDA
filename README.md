# Breadth First Search and Binary Tree algorithms in CUDA 
CUDA Programming: Bidirectional Breadth-first search over graphs with 14,000,000+ nodes and 34,000,000 arcs. Also included binary tree algorithm

**Professor: Dr. Doina Bein**

CPSC 479 High Performance Computing for Data Science - Spring 2018 - California State University Fullerton

Link to PPT slides: https://drive.google.com/open?id=1yvSYNQWPRfprk_XCQTeYhMUTpJ2aGqMDaw21bC6uxhc

Link to Report: https://drive.google.com/open?id=1Ygq1SgPNy9FugePJaf9gByz6i-srwpntzcKVZjrpf1c

## Prerequisites and Preprocessing instructions
1. A good GPU, in this project we used Nvidia GTX 1080 Ti
2. Ubuntu 16.04 with CUDA
3. Download Central USA and NYC distance and travel graphs via link:
http://www.dis.uniroma1.it/challenge9/download.shtml
4. Preprocess data so that only the first two columns (warning: large datasets with 14,000,000+ nodes and 34,000,000+ arcs) remained

## Commands to execute bidirectional BFS program
```
# nvcc bfsearch.cu
# .a.out NYCdistance.txt
```
If you want to process larger graphs such as Full USA and Central USA, you have to preprocess those graphs first.

## Commands to execute binary tree algorithm
```
# cd binary_tree
# make
# ./main
```


