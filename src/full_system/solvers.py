import numpy as np

def numerical_fv(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, q_0):
    """Takes initial state tensor of shape (N,) as input and propagates"""

    N = h_0.shape[0]
    g_prime = g*(rho_w-rho)/rho_w
    A = 2*alpha*np.sqrt(g/g_prime)
    B = (6*nu*q_0/g)**(1/3)
    C = (g/g_prime)**(1/6)

    # Convert t to dimensionless variables
    t_final = t_final * q_0/(2*B**2*C**5)

    j = np.arange(0,N)
    chi_plus = (j+1)/N
    chi_minus = j/N

    h_list = []
    h = h_0
    h_list.append(h.reshape(-1,1))

    x_G_list = []
    x_G = x_G_0
    x_G_list.append(x_G)

    def G_plus(h):
        h_plus = np.concatenate([h[1:],-2*h[-1].reshape(1)], axis=0)
        return chi_plus*N*(h_plus-h)
    
    def G_minus(h):
        h_minus = np.concatenate([np.zeros(1), h[:-1]], axis=0)
        return chi_minus*N*(h-h_minus)
    
    def F_plus(h):
        h_plus = np.concatenate([h[1:],-2*h[-1].reshape(1)], axis=0)
        return (0.5*(h_plus+h)+A*chi_plus)**3*N*(h_plus-h)
    
    def F_minus(h):
        h_minus = np.concatenate([np.zeros(1), h[:-1]], axis=0)
        return (0.5*(h+h_minus)+A*chi_minus)**3*N*(h-h_minus)
    
    def x_G_dot(h, x_G):
        term_1 = 1.5*(A*x_G)**2*(2*N*x_G*h[-1]-1/(2*N*x_G*h[-1]))
        term_2 = 2*A**2*x_G*N*h[-1]
        return np.min([term_1, term_2])
    
    delta_t = t_final/(num_steps*num_microsteps)

    for i in range(num_steps):
        for ii in range(num_microsteps):
            dx_G_dt = x_G_dot(h, x_G)
            dh_dt = dx_G_dt*N/x_G*(G_plus(h)-G_minus(h))+N/(x_G**2)*(F_plus(h)-F_minus(h))
            x_G = x_G + dx_G_dt*delta_t
            h = h + dh_dt*delta_t
        x_G_list.append(x_G)
        h_list.append(h.reshape(-1,1))

    # Concat into tensors and scale back to dimensional space
    h_tnsr = B*C*np.concatenate(h_list, axis=1)
    x_G_tnsr = 2*B*C**4*np.array(x_G).reshape(1,-1)
    chi = np.linspace(0.5*N, 1-0.5*N, num=N)
    x_tnsr = chi*x_G_tnsr

    return x_tnsr, h_tnsr