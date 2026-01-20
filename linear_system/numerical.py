import numpy as np

def compute_numerical(h_0, num_microsteps, num_t_steps, t_final, dim, L, h_G, a, k):
    """Computes numerical solution for initial problem with h_0(x)=-h_G*x/L + 2*h_G

        Parameters:
        - h_0 (function): Function for initial state of h at t=0
        - num_microsteps (int): Number of steps to compute per time step (true time step=num_t_steps*num_microsteps)
        - num_t_steps (int): Number of time steps to compute solution at
        - t_final (int): end time for solution to be computed to
        - dim (int): Number of spatial grid points to use
        - L (float): Domain width -> [0, L]
        - h_G (float): height of fixed boundary at x=L
        - a (float): Linear source term in PDE
        - k (float): Diffusion coefficient

        Returns h which is a numpy array of shape (dim, num_t_steps, 1) representing the solution on the specified domain in space and time.
    """
    print("Calculating numerical solution...")
    w = L/dim

    x = np.linspace(0+0.5*w,L-0.5*w, num=dim).reshape(-1, 1)
    t = np.linspace(0,t_final, num=num_t_steps).reshape(1,-1)

    delta_t = (t[0,1]-t[0,0])/num_microsteps
    converge = True if delta_t<=(w**2/(2*k)) else False
    print(f"w^2/2k: {w**2/(2*k):.04f} | delta_t: {delta_t:.04f} | -> Expected to converge: {converge}")

    A = np.eye(dim, k=1)+np.eye(dim, k=-1)-2*np.eye(dim)
    A[0,0] = -1
    A[-1,-1] = -3
    A = A*k/w**2

    B = a*np.ones(dim)
    B[-1]+= 2*h_G*k/w**2
    B = B.reshape(-1,1)

    h_list = []
    h_i = h_0(x)
    h_list.append(h_i)

    for i in range(num_t_steps):
        for ii in range(num_microsteps):
            h_i = h_i + (A@h_i+B)*delta_t
        h_list.append(h_i)

    h = np.concatenate(h_list, axis=1)
    print("Complete.")
    return h