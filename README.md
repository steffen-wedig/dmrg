# Density Matrix Renormalization Group


## Overview

This repo contains an implemetation of the denstiy matrix renormalization group algorithm in its modern MPO/MPS version. DMRG is a quantum chemistry algorithm that calculates the energy of 1 dimensional, correlated systems. 

In this project, we implemented the algorithm for the transverse Ising model and the molecular Hamiltonian in second quantization. The molecular Hamiltonian is still work in progress, due to  challenging construction of the local MPO matrices.


## Installation 

Clone the repository and in a python 3.11 conda environment, run:
```
pip install ./dmrg
```