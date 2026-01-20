import numpy as np

def compute_analytical(num_fourier_terms, num_t_steps, t_final, dim, L, h_G, a, k):
    """Computes analytical solution for initial problem with h_0(x)=-h_G*x/L + 2*h_G

        Parameters:
        - num_fourier_terms (int): Number of Fourier terms in solution
        - num_t_steps (int): Number of time steps to compute solution at
        - t_final (int): end time for solution to be computed to
        - dim (int): Number of spatial grid points to use
        - L (float): Domain width -> [0, L]
        - h_G (float): height of fixed boundary at x=L
        - a (float): Linear source term in PDE
        - k (float): Diffusion coefficient

        Returns h which is a numpy array of shape (dim, num_t_steps, 1) representing the solution on the specified domain in space and time.
    """
    print("Calculating analytical solution...")
    pi = np.pi
    w = L/dim

    x = np.linspace(0+0.5*w,L-0.5*w, num=dim).reshape(-1, 1, 1)
    t = np.linspace(0,t_final, num=num_t_steps).reshape(1,-1, 1)
    n = np.arange(0,num_fourier_terms, 1).reshape(1,1,-1)

    def A(n):
        f_1 = 8*h_G/(pi**2*(2*n+1)**2)
        f_2 = -16*a*L**2*(-1)**n/(k*pi**3*(2*n+1)**3)
        return f_1 + f_2

    A_n = A(n)

    # Compute fourier terms and sum them up:
    h_3_tnsr = np.exp(-pi**2*(2*n+1)**2*k*t/(4*L**2))*A_n*np.cos(pi*x*(2*n+1)/(2*L))
    h_f_terms = h_3_tnsr.sum(axis=2).reshape(h_3_tnsr.shape[0], h_3_tnsr.shape[1], 1)

    # Add to rest of terms to give final solution:
    h = h_f_terms - a*x**2/(2*k)+h_G+a*L**2/(2*k)
    print("Complete.")
    return h