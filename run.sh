#!/bin/bash
# Isaac Lehman ~ COMP322 ~ Laplace Heat Distribution
# Bash script for exicuting mpirun on 4-16 processes

# COMPILE
mpicc -openmp -lm -o J.exe Jacobi.c

# SYNC the nodes
bccd-syncdir . ~/machines-openmpi

# RUN file and pipe to output.csv file
for i in {4..16..4}
do
	mpirun -machinefile ~/machines-openmpi -np $i --bynode /tmp/node000-bccd/J.exe .01 500000 1
done
