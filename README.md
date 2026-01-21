# Modelling Viscous Ice Sheet Flow
My bachelor's project for my physics degree at Oxford University, working under the supervision of Dr. Andrew Wells.

## Linear System
At present, you can find analytical and numerical solutions to the following problem on $x\in[0,L]$ at the directory `src/linear_system/`:

$$\frac{\partial h}{\partial t} = k\frac{\partial^2h}{\partial^2 x}+a$$

With the following boundary conditions:

$$\frac{\partial h}{\partial x}(0,t) = 0$$
$$h(L,t) = h_G$$
$$h(x,0) = -\frac{h_G}{L}+2h_G$$

This is just a simple linear system, and I plan for the non-linear case to follow.
