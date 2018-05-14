# Breadth First Search and Binary Tree algorithms in CUDA 
CUDA Programming: Bidirectional breadth-first search (BFS) for graphs with 14,000,000+ nodes and 34,000,000+ arcs. Also included basic binary tree algorithm (construction + traversal).

**Phillip Ly**

**Professor: Dr. Doina Bein**

CPSC 479 High Performance Computing for Data Science - Spring 2018 - California State University Fullerton

Link to PPT slides: https://drive.google.com/open?id=1yvSYNQWPRfprk_XCQTeYhMUTpJ2aGqMDaw21bC6uxhc

Link to Report: https://drive.google.com/open?id=1Ygq1SgPNy9FugePJaf9gByz6i-srwpntzcKVZjrpf1c

## Prerequisites and Preprocessing instructions
1. A good GPU, in this project we used Nvidia GTX 1080 Ti
2. Ubuntu 16.04 with CUDA. In this project, our server is equipped with six NVIDIA GeForce GTX 1080 Ti (12GB of VRAM). Furthermore, this Ubuntu server also includes 40 Intel Xeon processors E5-2630 v4 2.20GHz which have 2 threads per core, 10 cores per socket, and 256 GB of main memory. 
3. Download Central USA and NYC distance and travel graphs via link:
http://www.dis.uniroma1.it/challenge9/download.shtml
4. Preprocess data so that only the first two columns (warning: large datasets with 14,000,000+ nodes and 34,000,000+ arcs) remained

## Commands to execute bidirectional BFS program
For example, in order to process New York City's distance graph:
```
# cd project2
# nvcc bfsearch.cu
# ./a.out NYCdistance.txt
```
If you want to process larger graphs such as Full USA and Central USA, you have to preprocess those graphs first.

## Commands to run binary tree algorithm
```
# cd project2
# cd binary_tree
# make
# ./main
```


