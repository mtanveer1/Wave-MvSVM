# Wave-MvSVM: Enhancing Multiview Synergy: Robust Learning by Exploiting the Wave Loss Function with Consensus and Complementarity Principles

Please cite the following paper if you are using this code. Reference: Quadir, A., Akhtar, M., & Tanveer, M. (2025). Enhancing multiview synergy: Robust learning by exploiting the wave loss function with consensus and complementarity principles. Neural Networks, 188, 107433. https://doi.org/10.1016/j.neunet.2025.107433


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
The experiments are executed on a computing system possessing Matlab R2023a software, an Intel(R) Xeon(R) CPU E5-2697 v4 processor operating at 2.30 GHz with 128-GB Random Access Memory (RAM), and a Windows-11 operating platform.

We have put a demo of the “Wave-MvSVM” model.

Description of files: 
main.m: This is the main file to run selected algorithms on datasets. In the path variable specificy the path to the folder containing the codes and datasets on which you wish to run the algorithm. 
ADMM.m: Solving the optimization problem.
GD.m: Update the parameter \zeta_1 and \zeta_2.
predict_svm2.m: Function to evaluate the accuracy.
AUC.m: Function to evaluate the AUC.
maxview.m: Function to evaluate the maxview.
kernel.m: Evaluate the kernel matrix.

For the detailed experimental setup, please follow the paper. If you find any bugs/issues, please write to A. Quadir (mscphd2207141002@iiti.ac.in) and Mushir Akhtar (phd2101241004@iiti.ac.in).

