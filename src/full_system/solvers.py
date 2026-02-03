import numpy as np
import time

def numerical_fv(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, q_0):
    """Takes initial state tensor of shape (N,) as input and propagates"""

    def F_plus(h):
        h_plus = np.concatenate([h[1:],np.zeros(1)], axis=0)
        h_term_1 = h_plus + h
        h_term_1[-1] = 0
        h_term_2 = h_plus - h
        h_term_2[-1] = -2*h[-1]
        return (0.5*h_term_1 + A*chi_plus)**3*N*h_term_2    
    def F_minus(h, x_G):
        h_minus = np.concatenate([np.zeros(1), h[:-1]], axis=0)
        boundary_constant = np.zeros(N)
        boundary_constant[0] = -x_G
        first_term = (0.5*(h+h_minus)+A*chi_minus)**3*N*(h-h_minus)
        first_term[0]=0
        return first_term + boundary_constant
    
    def x_G_dot(h, x_G):
        #time.sleep(1)
        term_1 = 1.5*(A*x_G)**2*(2*N*h[-1]/x_G-x_G/(2*N*h[-1]))
        term_2 = 2*A**2*x_G*N*h[-1]
        return np.min([term_1, term_2])
    
    def advective_term(h, x_G, dx_G_dt):
        h_minus = np.concatenate([np.zeros(1), h[:-1]], axis=0)
        term_1 = (h-h_minus)
        term_1[0] = 2*x_G/((h[1]-3*h[0])*N)
        return N*dx_G_dt*chi*term_1/x_G
    
    h_list = []
    h = h_0
    x_G_list = []
    x_G = x_G_0
    H_G_list = []
    q_G_list = []
    x_G_dot_list = []
    t_G_list = []

    def compute_shelf(h, t, x_G_list, H_G_list, q_G_list, x_G_dot_list, t_G_list):
        x_list = []
        x_G_arr = np.array(x_G_list)
        H_G_arr = np.array(H_G_list)
        q_G_arr = np.array(q_G_list)
        x_G_dot_arr = np.array(x_G_dot_list)
        t_G_arr = np.array(t_G_list)

        for idx in range(x_G.shape[0]):
            x_i = x_G_arr[i]+np.sum(delta_t*q_G_arr[i:]/H_G_arr[i:])+(g_prime/(8*nu))*np.sum(delta_t*(t-t_G_arr[i:])*(q_G_arr[i:]-H_G_arr[i:]*x_G_dot_arr[i:]))
            x_list.append(x_i)
        x_arr = np.array(x_list)
        H_arr = 8*nu*H_G_arr/(g_prime*H_G_arr*(t-t_G_arr)+8*nu)
        return x_arr, H_arr
        
    
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
    chi = (j+0.5)/N

    # Create lists to hold variables at each time step
    h_list = []
    h = h_0
    x_G_list = []
    x_G = x_G_0
    H_G_list = []
    q_G_list = []
    x_G_dot_list = []
    t_G_list = []

    # Append t=0 values to lists
    h_list.append(h.reshape(-1,1))
    x_G_list.append(x_G)
    x_G_dot_list.append(x_G_dot(h, x_G))
    t_G_list.append(0)
    H_G = A*x_G
    H_G_list.append(H_G)
    q_G_list.append(-g/(3*nu)*H_G**3*(-2*h[-1]*N/x_G))

    delta_t = t_final/(num_steps*num_microsteps)

    for i in range(num_steps):
        # Calculate shelf:


        for ii in range(num_microsteps):
            dx_G_dt = x_G_dot(h, x_G)
            # print(f"x_G:{ x_G}")
            # time.sleep(1)
            dh_dt = advective_term(h, x_G, dx_G_dt)+N/(x_G**2)*(F_plus(h)-F_minus(h, x_G))
            x_G = x_G + dx_G_dt*delta_t
            h = h + dh_dt*delta_t
        
        x_G_list.append(x_G)
        h_list.append(h.reshape(-1,1))
        x_G_dot_list.append(x_G_dot(h, x_G))
        t_G_list.append(0)
        H_G = A*x_G
        H_G_list.append(H_G)
        q_G_list.append(-g/(3*nu)*H_G**3*(-2*h[-1]*N/x_G))
        t_G_list.append((i+1)*num_microsteps*delta_t)

    
    #print(x_G_list)
    print(2*B*C**4)
    print(max(x_G_list))
    # Concat into tensors and scale back to dimensional space
    h_tnsr = B*C*np.concatenate(h_list, axis=1)
    x_G_tnsr = 2*B*C**4*np.array(x_G_list).reshape(1,-1)
    chi = np.linspace(0.5/N, 1-0.5/N, num=N).reshape(-1,1)
    x_tnsr = chi*x_G_tnsr

    print(x_tnsr)
    return x_tnsr, h_tnsr