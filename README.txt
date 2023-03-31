SVM-Kernel-Selection
====================
Some code from for a project during my Masters degree implementing a few kernel selection routines from the paper
"Learning the kernel matrix with semidefinite programming." by Grigoris R. Lanckriet, Nello Cristianini, Peter Bartlett, Leonid E. Ghaoui, and Michael I. Jordan.
from the Journal of Machine learning research, 5(Jan), 27-72.

Overview of Modules: 
NOTE: modules typically executed in the order of top down, with respect to how I have them listed below.
e.g.    -get your data
        -create a bunch of different kernels
        -Find the best kernel via a Quadratic Program
        -run a version of an SVM
        -get error measures )


getdata.py
===============
Module for reading your data files, cleaning it, and separating it into training/testing sets.
Currently holds functions that allow you to call for cleaned versions of particular data sets.


Kernels.py
===============
Module which does the following for data inputs:
1) Creates Kernels from data. Can create Linear, Gaussian, Polynomial, T-Student, Cauchy, Wave, Power, or Sigmoid Kernels
2) Centers the Kernel Matrices, acting to center the data within the kernel induced feature space
3) Performs PCA in the kernel feature space via an eigendecomposition to retrieve reduced component transformations.
4) Normalizes the transformed components.
5) Returns list of kernel matrices


OptimalKernels.py
==============
Module contains two choices for optimzaton routines that select the best convex combination of kernels generated with Kernels.py
1) OptimalDualK_CSVM:
        Single stage kernel selection SVM routine. Optimizes traditional soft-margin SVM by selecting optimal convex combinations
        of kernels
2) max_align:
        First stage of a two stage kernel selection SVM routine. This stage finds the best convex combination of kernels that serves
        to optimize the 'alighnment' of the combination kernel with an 'ideal' kernel, defined by the outer product of trianing label vectors.
        Returns optimal Kernel to use in second stage of routine. Second stage is balanced-softmargin-SVM with optimal kernel.


C-SVM.py
==============
Module contains the kernalized primal and dual formulation of the balanced-softmargin-SVM.


GetResults.py
==============
Contains functions that recieve linear prediction function from C-SVM routines and return AUC,MCC,%Error.
Optional piece of matplotlib code allows you to compare predicted labels vs. actual labels in training data.


RF.py
=============
Experiment RunFile to run full routines, do k-fold cross-validation, plot error results, plot tranformed features, etc

