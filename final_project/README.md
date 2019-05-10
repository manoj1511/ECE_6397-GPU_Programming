# STRUCTURE TENSOR TRACTOGRAPHY

All instructions are specific to the bridges supercomputer housed at the Pittsburgh Supercomputing Center.
Bridges User Guide -> https://www.psc.edu/bridges/user-guide

## Compiling
Please use the make file attached to compile the code.

### Requirements:
1. Cuda Toolkit
2. Eigen module
3. GCC compiler

All the requirements are present in Bridges.

### Procudure
1. $ module load cuda
2. $ module load eigen
3. To compile cpu version run
   
   $ make cpu
   
   To compile gpu version run
   
   $ make gpu

## Running
1. create an empty folder caled Trace in your working directory so that cpu version can write the trace files in it.
2. Get an interactive gpu node.(Recommended)
   
   $ interact -gpu
   
   You can run both cpu and gpu coode here. Slurm will allocate a single Nvidia P100 GPU node. The GPU node also comes with 
   haswell cpu with 16 cores.
3. To run CPU code
   
   $ ./cpu
   
   To run GPU code
   
   $ ./gpu
4. You can copy the data from Trace back to your CPU and visualize using the provided python code.
