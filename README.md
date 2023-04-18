# M2_generative_hopfield_network
Hopfield Networks and Modified Variants

This repository contains Python code for Hopfield networks and their modified variants, including rHN_0, rHN_S, and rHN_G. These networks are described in the paper "Transformations in the scale of behavior and the global optimization of constraints in adaptive networks" by Richard A. Watson, C. L. Buckley, and Rob Mills.

Overview
This repository includes four Python scripts:

1. hopfield.py: This script implements the original Hopfield network and provides methods for training and running simulations.

2. rhn_0.py: This script implements rHN_0, a modified Hopfield network that uses a fixed-point iteration for constraint satisfaction problems.

3. rhn_s.py: This script implements rHN_S, a modified Hopfield network that updates the test part of the generate-and-test process by using a modified energy function accounting for learned weights.

4. rhn_g.py: This script implements rHN_G, a modified Hopfield network that updates the test part of the generate-and-test process by using a modified variation operator.

In addition, the results folder contains statistics and results from running simulations with the implemented networks.

Requirements
The code was written and tested with Python 3.7.9, and requires the following libraries to be installed:

NumPy
Matplotlib
Usage

To run the simulations, simply run the corresponding Python scripts in a terminal or in an IDE such as Jupyter Notebook. The scripts will output plots and statistics similar to those shown in the paper.

The output of the scripts will be printed to the console and/or saved to files in the results folder.

Acknowledgements
The code in this repository was created and is being updated by [Your Name] as part of [Your Project Name] at [Your Institution] with [Your Professor's Name].

The code is intended for researchers and enthusiasts in the field of complex adaptive systems and provides a simple and easily understandable implementation of Hopfield networks and their modified variants. The project is ongoing and may be updated in the future to include additional experiments or improvements to the existing code.

Citation
If you use this code for research purposes, please cite the following paper:

Watson, R. A., Buckley, C. L., & Mills, R. (2018). Optimization in "Self-Modeling" Complex Adaptive Systems. https://onlinelibrary.wiley.com/doi/abs/10.1002/cplx.20346

Watson, R. A., Mills, R., & Buckley, C. L. (2011). Transformations in the scale of behavior and the global optimization of constraints in adaptive networks. Adaptive Behavior, 19(5), 343-359. https://doi.org/10.1177/1059712311412797
