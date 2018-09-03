# Lid Driven Cavity solver based on Finite Pointset Method (FPM)

This implements a fluid flow solver based on the meshfree Finite Pointset Method. The derivatives are found using the Moving Least Squares approach and the Poissons equation is solved using the GMRES Krylov iterative method. The default settings generate close to 1000 points and Re = 100. 
