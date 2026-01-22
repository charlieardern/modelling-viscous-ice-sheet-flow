import numpy as np

def A(s, b, s_G):
    # s has dimensions (dim, 1)
    s1 = 2*s_G-s[-1].reshape(1,1)
    t1 = np.concatenate([s[1:], s1], axis=0)

    b2 = b[-1].reshape(1,1)
    t2 = np.concatenate([b[1:], b2], axis=0)
    
    s3 = 2*s_G-s[-2:][::-1].reshape(2,1)
    t3 =  np.concatenate([s[2:], s3], axis=0)
    return (t1-t2)**3*(t3-s)

def B(s,b):
    s1 = s[0].reshape(1,1)
    t1 = np.concatenate([s1,s[:-1]], axis=0)

    b2 = b[0].reshape(1,1)
    t2 = np.concatenate([b2,b[:-1]], axis=0)

    s3 = s[:2][::-1].reshape(2,1)
    t3 = np.concatenate([s3, s[:-2]], axis=0)
    return (t1-t2)**3*(s-t3)

def F_plus(s, b, s_G, w):
    s1 = 2*s_G-s[-1].reshape(1,1)
    t1 = np.concatenate([s[1:], s1], axis=0)

    b2 = b[-1].reshape(1,1)
    t2 = np.concatenate([b[1:], b2], axis=0)

    return (s+t1+b+t2)**3*(t1-s)/(8*w)

def F_minus(s, b, w):
    s1 = s[0].reshape(1,1)
    t1 = np.concatenate([s1,s[:-1]], axis=0)

    b2 = b[0].reshape(1,1)
    t2 = np.concatenate([b2,b[:-1]], axis=0)

    return (s + t1 + b + t2)**3*(s-t1)/(8*w)



def compute_numerical_explicit_fd(s_0, b, num_microsteps, num_t_steps, t_final, dim, L, s_G, a, g, eta):
    """Computes numerical solution for initial problem with h_0(x)=-h_G*x/L + 2*h_G with explicit time steps and finite difference method.

        Parameters:
        - s_0 (function): Function for initial state of s at t=0
        - b (numpy array): Bedrock shape
        - num_microsteps (int): Number of steps to compute per time step (true time step=num_t_steps*num_microsteps)
        - num_t_steps (int): Number of time steps to compute solution at
        - t_final (int): end time for solution to be computed to
        - dim (int): Number of spatial grid points to use
        - L (float): Domain width -> [0, L]
        - s_G (float): height of fixed boundary at x=L
        - a (float): Linear source term in PDE
        - g (float): gravitational field strength
        - eta (float): viscosity coefficient

        Returns s which is a numpy array of shape (dim, num_t_steps, 1) representing the solution on the specified domain in space and time.
    """
    print("Calculating explicit numerical solution...")
    w = L/dim

    x = np.linspace(0+0.5*w,L-0.5*w, num=dim).reshape(-1, 1)
    t = np.linspace(0,t_final, num=num_t_steps).reshape(1,-1)

    delta_t = (t[0,1]-t[0,0])/num_microsteps

    s_list = []

    s_i = s_0(x, s_G)
    s_list.append(s_i)

    for i in range(num_t_steps-1):
        for ii in range(num_microsteps):
            ds_dt = (g/(6*eta*w))*(A(s_i,b,s_G)-B(s_i,b))+a
            s_i = s_i+ds_dt*delta_t
        s_list.append(s_i)
    
    s = np.concatenate(s_list, axis=1)
    print("Complete.")
    return s

def compute_numerical_explicit_fv(s_0, b, num_microsteps, num_t_steps, t_final, dim, L, s_G, a, g, eta):
    """Computes numerical solution for initial problem with h_0(x)=-h_G*x/L + 2*h_G with explicit time steps and finite volume method.

        Parameters:
        - s_0 (function): Function for initial state of s at t=0
        - b (numpy array): Bedrock shape
        - num_microsteps (int): Number of steps to compute per time step (true time step=num_t_steps*num_microsteps)
        - num_t_steps (int): Number of time steps to compute solution at
        - t_final (int): end time for solution to be computed to
        - dim (int): Number of spatial grid points to use
        - L (float): Domain width -> [0, L]
        - s_G (float): height of fixed boundary at x=L
        - a (float): Linear source term in PDE
        - g (float): gravitational field strength
        - eta (float): viscosity coefficient

        Returns s which is a numpy array of shape (dim, num_t_steps, 1) representing the solution on the specified domain in space and time.
    """
    print("Calculating explicit numerical solution...")
    w = L/dim

    x = np.linspace(0+0.5*w,L-0.5*w, num=dim).reshape(-1, 1)
    t = np.linspace(0,t_final, num=num_t_steps).reshape(1,-1)

    delta_t = (t[0,1]-t[0,0])/num_microsteps

    s_list = []

    s_i = s_0(x, s_G)
    s_list.append(s_i)

    for i in range(num_t_steps-1):
        for ii in range(num_microsteps):
            ds_dt = (g/(3*eta*w))*(F_plus(s_i, b, s_G, w)-F_minus(s_i, b, w))+a
            s_i = s_i+ds_dt*delta_t
        s_list.append(s_i)
    
    s = np.concatenate(s_list, axis=1)
    print("Complete.")
    return s