import numpy as np
from tqdm.auto import tqdm
from scipy.optimize import root_scalar

def numerical_fv(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=False, hide_output=False, sheet_accumulation=True):
    """Takes initial state tensor of shape (N,) as input and propagates"""

    def F_plus(h, x_G):
        h_plus = np.concatenate([h[1:],np.zeros(1)], axis=0)
        h_term_1 = h_plus + h
        h_term_1[-1] = 2*eps*x_G
        h_term_2 = h_plus - h
        h_term_2[-1] = (8*eps*x_G+h[-2]-9*h[-1])/3
        return (0.5*h_term_1 + A*x_G*chi_plus)**3*N*h_term_2    
    
    def F_minus(h, x_G):
        h_minus = np.concatenate([np.zeros(1), h[:-1]], axis=0)
        boundary_constant = np.zeros(N)
        boundary_constant[0] = -x_G
        first_term = (0.5*(h+h_minus)+A*x_G*chi_minus)**3*N*(h-h_minus)
        first_term[0]=0
        return first_term + boundary_constant
    
    def x_G_dot(h, x_G):
        dh_dchi = (8*eps*x_G+h[-2]-9*h[-1])*N/3
        term_1 = 1.5*(eps+A)**2*x_G*(dh_dchi**2-x_G**2)/(eps*x_G-dh_dchi)
        term_2 = -(eps+A)**2*x_G*dh_dchi
        return np.min([term_1, term_2])
    
    def advective_term(h, x_G, dx_G_dt):

        # Equation to be solved to find h_{-1}
        def f(h_neg): return (0.5*(h[0]+h_neg))**3*(h[0]-h_neg)*N+x_G

        if dx_G_dt > 0:
            h_minus = np.concatenate([np.zeros(1), h[:-1]], axis=0)
            h_minus_minus = np.concatenate([np.zeros(2), h[:-2]], axis=0)
            try:
                h_neg = root_scalar(f, bracket=[0, 5], method="brentq").root # h_{-1}
            except:
                h_neg = np.nan

            term_1 = 0.5*(3*h-4*h_minus+h_minus_minus)
            term_1[0] = 0.5*(h[1]-h_neg)
            term_1[1] = 0.5*(3*h[1]-4*h[0]+h_neg)
        
        else:
            h_plus = np.concatenate([h[1:], np.zeros(1)], axis=0)
            h_plus_plus = np.concatenate([h[2:], np.zeros(2)], axis=0)
            h_N = (8*eps*x_G-6*h[-1]+h[-2])/3
            term_1 = 0.5*(-3*h+4*h_plus-h_plus_plus)
            term_1[-1] = 0.5*(h_N-h[-2])
            term_1[-2] = 0.5*(-3*h[-2]+4*h[-1]-h_N)
        
        return N*dx_G_dt*chi*term_1/x_G
    
    h_list = []
    h = h_0
    x_G_list = []
    x_G = x_G_0
    H_G_list = []
    q_G_list = []
    x_G_dot_list = []
    t_G_list = []

    def compute_shelf(t, num_microsteps, x_G_list, H_G_list, q_G_list, x_G_dot_list, t_G_list):
        x_list = []
        x_G_arr = 2*B*C**4*np.array(x_G_list)
        H_G_arr = np.array(H_G_list)*B*C
        q_G_arr = np.array(q_G_list)*B**3/2
        x_G_dot_arr = np.array(x_G_dot_list)*q_0/(B*C)
        t_G_arr = np.array(t_G_list)*2*B**2*C**5/(a*L)
        dt_scaled = delta_t*2*B**2*C**5/(a*L)
        t_scaled = t*2*B**2*C**5/(a*L)

        for iii in range(1+round((len(x_G_list))/num_microsteps)):
            idx = num_microsteps*iii
            term_1 = np.sum(dt_scaled*q_G_arr[idx:]/H_G_arr[idx:])
            term_2 = (g_prime/(8*nu))*np.sum(dt_scaled*(t_scaled-t_G_arr[idx:])*(q_G_arr[idx:]-H_G_arr[idx:]*x_G_dot_arr[idx:]))
            x_i = x_G_arr[idx]+term_1+term_2
            x_list.append(x_i)
        x_arr = np.array(x_list)
        H_arr = 8*nu*H_G_arr[::num_microsteps]/(g_prime*H_G_arr[::num_microsteps]*(t_scaled-t_G_arr[::num_microsteps])+8*nu)
        return x_arr, H_arr
        
    q_0 = a*L
    N = h_0.shape[0]
    g_prime = g*(rho_w-rho)/rho_w
    A = 2*alpha*np.sqrt(g/g_prime)
    B = (6*nu*q_0/g)**(1/3)
    C = (g/g_prime)**(1/6)
    eps = 2*alpha*(g_prime/g)**(0.5)*rho_w/rho
    D = 2*B*C**4/L

    # Convert t to dimensionless variables
    t_final = t_final * q_0/(2*B**2*C**5)

    j = np.arange(0,N)
    chi_plus = (j+1)/N
    chi_minus = j/N
    chi = (j+0.5)/N

    h = h_0
    x_G = x_G_0

    # Lists for final results
    h_list = []
    x_G_list = []

    # Create lists to hold variables at each time step
    h_full_list = []
    x_G_full_list = []
    H_G_list = []
    q_G_list = []
    x_G_dot_list = []
    t_G_list = []

    # Append t=0 values to lists
    h_list.append(h.reshape(-1,1))
    x_G_list.append(x_G)
    x_G_full_list.append(x_G)
    x_G_dot_list.append(x_G_dot(h, x_G))
    t_G_list.append(0)
    H_G = A*x_G + eps*x_G
    H_G_list.append(H_G)
    q_G_list.append(-g/(3*nu)*H_G**3*((8*eps*x_G+h[-2]-9*h[-1])*N/(3*x_G)))

    delta_t = t_final/(num_steps*num_microsteps)

    x_shelf_list = []
    H_shelf_list = []

    diverged = False
    for i in tqdm(range(num_steps), disable=hide_output):
        # Calculate shelf:
        t = i*num_microsteps*delta_t
        if not test_mode:
            x_shelf, H_shelf = compute_shelf(t, num_microsteps, x_G_full_list, H_G_list, q_G_list, x_G_dot_list, t_G_list)
            x_shelf_list.append(np.pad(x_shelf, (0, 100-len(x_shelf)), mode='edge').reshape(-1,1))
            H_shelf_list.append(np.pad(H_shelf, (0, 100-len(H_shelf)), mode='edge').reshape(-1,1))

        # Calculate sheet:
        for ii in range(num_microsteps):
            dx_G_dt = x_G_dot(h, x_G)
            dh_dt = advective_term(h, x_G, dx_G_dt)+N/(x_G**2)*(F_plus(h, x_G)-F_minus(h, x_G)) + D*int(sheet_accumulation)
            x_G = x_G + dx_G_dt*delta_t
            h = h + dh_dt*delta_t

            x_G_full_list.append(x_G)
            h_full_list.append(h)
            x_G_dot_list.append(x_G_dot(h, x_G))
            H_G = A*x_G + eps*x_G
            H_G_list.append(H_G)
            q_G_list.append(-g/(3*nu)*H_G**3*((8*eps*x_G+h[-2]-9*h[-1])*N/(3*x_G)))
            t_G_list.append(i*num_microsteps*delta_t+(ii+1)*delta_t)

            if np.isnan(np.array(h)).any() and not diverged:
                diverged = True
                if not hide_output:
                    print("Solution has diverged.")
                return False, np.nan

        h_list.append(h.reshape(-1,1))
        x_G_list.append(x_G)

    # Concat into tensors and scale back to dimensional space
    h_tnsr = B*C*np.concatenate(h_list, axis=1)
    x_G_tnsr = 2*B*C**4*np.array(x_G_list).reshape(1,-1)
    chi = np.linspace(0.5/N, 1-0.5/N, num=N).reshape(-1,1)
    x_tnsr = chi*x_G_tnsr
    if not test_mode:
        x_shelf_tnsr = np.concatenate(x_shelf_list, axis=1)
        H_shelf_tnsr = np.concatenate(H_shelf_list, axis=1)

    if np.isnan(h_tnsr).any():
        converge = False
    else:
        converge = True

    converge = not np.isnan(h_tnsr).any()
    if not hide_output:
        print(f"Converged: {converge}")

    if test_mode:
        out = x_tnsr, h_tnsr, converge, x_G_tnsr
    
    else:
        out = x_tnsr, h_tnsr, x_shelf_tnsr, H_shelf_tnsr

    return out

def numerical_fv_implicit(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=False, hide_output=False, sheet_accumulation=True):
    """Takes initial state tensor of shape (N,) as input and propagates"""
    
    def K_plus(h, x_G):
        h_plus = h[1:]
        h_i = h[:-1]
        #print(f"h_i.shape: {h_i.shape}")
        #print(f"h_minus.shape: {h_plus.shape}")
        chi_plus_slice = chi_plus[:-1]
        return (0.5*(h_i+h_plus)+A*x_G*chi_plus_slice)**3
    
    def K_minus(h, x_G):
        h_i = h[1:]
        h_minus = h[:-1]
        #print(f"h_i.shape: {h_i.shape}")
        #print(f"h_minus.shape: {h_minus.shape}")
        chi_minus_slice = chi_minus[1:]
        return (0.5*(h_i+h_minus)+A*x_G*chi_minus_slice)**3
    
    def x_G_dot(h, x_G):
        dh_dchi = (8*eps*x_G+h[-2]-9*h[-1])*N/3
        term_1 = 1.5*(eps+A)**2*x_G*(dh_dchi**2-x_G**2)/(eps*x_G-dh_dchi)
        term_2 = -(eps+A)**2*x_G*dh_dchi
        return np.min([term_1, term_2])
    
    def advective_term(h, x_G, dx_G_dt):

        # Equation to be solved to find h_{-1}
        def f(h_neg): return (0.5*(h[0]+h_neg))**3*(h[0]-h_neg)*N+x_G

        if dx_G_dt > 0:
            h_minus = np.concatenate([np.zeros(1), h[:-1]], axis=0)
            h_minus_minus = np.concatenate([np.zeros(2), h[:-2]], axis=0)
            try:
                h_neg = root_scalar(f, bracket=[0, 5], method="brentq").root # h_{-1}
            except:
                h_neg = np.nan

            term_1 = 0.5*(3*h-4*h_minus+h_minus_minus)
            term_1[0] = 0.5*(h[1]-h_neg)
            term_1[1] = 0.5*(3*h[1]-4*h[0]+h_neg)
        
        else:
            h_plus = np.concatenate([h[1:], np.zeros(1)], axis=0)
            h_plus_plus = np.concatenate([h[2:], np.zeros(2)], axis=0)
            h_N = (8*eps*x_G-6*h[-1]+h[-2])/3
            term_1 = 0.5*(-3*h+4*h_plus-h_plus_plus)
            term_1[-1] = 0.5*(h_N-h[-2])
            term_1[-2] = 0.5*(-3*h[-2]+4*h[-1]-h_N)
        
        return N*dx_G_dt*chi*term_1/x_G
    
    h_list = []
    h = h_0
    x_G_list = []
    x_G = x_G_0
    H_G_list = []
    q_G_list = []
    x_G_dot_list = []
    t_G_list = []

    def compute_shelf(t, num_microsteps, x_G_list, H_G_list, q_G_list, x_G_dot_list, t_G_list):
        x_list = []
        x_G_arr = 2*B*C**4*np.array(x_G_list)
        H_G_arr = np.array(H_G_list)*B*C
        q_G_arr = np.array(q_G_list)*B**3/2
        x_G_dot_arr = np.array(x_G_dot_list)*q_0/(B*C)
        t_G_arr = np.array(t_G_list)*2*B**2*C**5/(a*L)
        dt_scaled = delta_t*2*B**2*C**5/(a*L)
        t_scaled = t*2*B**2*C**5/(a*L)

        for iii in range(1+round((len(x_G_list))/num_microsteps)):
            idx = num_microsteps*iii
            term_1 = np.sum(dt_scaled*q_G_arr[idx:]/H_G_arr[idx:])
            term_2 = (g_prime/(8*nu))*np.sum(dt_scaled*(t_scaled-t_G_arr[idx:])*(q_G_arr[idx:]-H_G_arr[idx:]*x_G_dot_arr[idx:]))
            x_i = x_G_arr[idx]+term_1+term_2
            x_list.append(x_i)
        x_arr = np.array(x_list)
        H_arr = 8*nu*H_G_arr[::num_microsteps]/(g_prime*H_G_arr[::num_microsteps]*(t_scaled-t_G_arr[::num_microsteps])+8*nu)
        return x_arr, H_arr
        
    q_0 = a*L
    N = h_0.shape[0]
    g_prime = g*(rho_w-rho)/rho_w
    A = 2*alpha*np.sqrt(g/g_prime)
    B = (6*nu*q_0/g)**(1/3)
    C = (g/g_prime)**(1/6)
    eps = 2*alpha*(g_prime/g)**(0.5)*rho_w/rho
    D = 2*B*C**4/L

    # Convert t to dimensionless variables
    t_final = t_final * q_0/(2*B**2*C**5)

    j = np.arange(0,N)
    chi_plus = (j+1)/N
    chi_minus = j/N
    chi = (j+0.5)/N

    h = h_0
    x_G = x_G_0

    # Lists for final results
    h_list = []
    x_G_list = []

    # Create lists to hold variables at each time step
    h_full_list = []
    x_G_full_list = []
    H_G_list = []
    q_G_list = []
    x_G_dot_list = []
    t_G_list = []

    # Append t=0 values to lists
    h_list.append(h.reshape(-1,1))
    x_G_list.append(x_G)
    x_G_full_list.append(x_G)
    x_G_dot_list.append(x_G_dot(h, x_G))
    t_G_list.append(0)
    H_G = A*x_G + eps*x_G
    H_G_list.append(H_G)
    q_G_list.append(-g/(3*nu)*H_G**3*((8*eps*x_G+h[-2]-9*h[-1])*N/(3*x_G)))

    delta_t = t_final/(num_steps*num_microsteps)

    x_shelf_list = []
    H_shelf_list = []

    diverged = False
    for i in tqdm(range(num_steps), disable=hide_output):
        # Calculate shelf:
        t = i*num_microsteps*delta_t
        if not test_mode:
            x_shelf, H_shelf = compute_shelf(t, num_microsteps, x_G_full_list, H_G_list, q_G_list, x_G_dot_list, t_G_list)
            x_shelf_list.append(np.pad(x_shelf, (0, 100-len(x_shelf)), mode='edge').reshape(-1,1))
            H_shelf_list.append(np.pad(H_shelf, (0, 100-len(H_shelf)), mode='edge').reshape(-1,1))

        # Calculate sheet:
        for ii in range(num_microsteps):
            dx_G_dt = x_G_dot(h, x_G)
            k_plus = K_plus(h, x_G)
            k_minus = K_minus(h, x_G)
            C_n = delta_t*N**2/(x_G**2)
            B_n = (eps+A)**3*x_G**3/3

            # Construct matrix
            m = -C_n*k_minus
            m_prime = 1+C_n*(k_plus[1:]+k_minus[:-1])
            m_pprime = -C_n*k_plus

            lower = np.diag(m) # shape (N-1, N-1)
            lower[-1,-1] = -C_n*(B_n+k_minus[-1])
            lower = np.pad(lower, ((1, 0), (0, 1)), mode='constant') # shape (N, N)

            diagonal = np.diag(m_prime) # shape (N-2, N-2)
            diagonal = np.pad(diagonal, ((1, 1), (1, 1)), mode='constant') # shape (N, N)
            diagonal[0,0] = 1+C_n*k_plus[0]
            diagonal[-1,-1] = 1 + C_n*(9*B_n+k_minus[-1])

            upper = np.diag(m_pprime) # shape (N-1, N-1)
            upper[0,0] = -C_n*k_plus[0]
            upper = np.pad(upper, ((0, 1), (1, 0)), mode='constant') # shape (N, N)
            
            # print(f"lower shape: {lower.shape}")
            # print(f"diagonal shape: {diagonal.shape}")
            # print(f"upper shape: {upper.shape}")

            M = lower + diagonal + upper

            #print(M-M.T)

            const_term = np.zeros(N)
            const_term[0] = C_n*x_G/N
            const_term[-1] = 8*C_n*B_n*eps*x_G

            adv = advective_term(h, x_G, dx_G_dt)

            h = np.linalg.solve(M, h+const_term+(adv+D*int(sheet_accumulation))*delta_t)

            x_G = x_G + dx_G_dt*delta_t
            x_G_full_list.append(x_G)
            h_full_list.append(h)
            x_G_dot_list.append(x_G_dot(h, x_G))
            H_G = A*x_G + eps*x_G
            H_G_list.append(H_G)
            q_G_list.append(-g/(3*nu)*H_G**3*((8*eps*x_G+h[-2]-9*h[-1])*N/(3*x_G)))
            t_G_list.append(i*num_microsteps*delta_t+(ii+1)*delta_t)

            if np.isnan(np.array(h)).any() and not diverged:
                diverged = True
                if not hide_output:
                    print("Solution has diverged.")
                return False, np.nan

        h_list.append(h.reshape(-1,1))
        x_G_list.append(x_G)

    # Concat into tensors and scale back to dimensional space
    h_tnsr = B*C*np.concatenate(h_list, axis=1)
    x_G_tnsr = 2*B*C**4*np.array(x_G_list).reshape(1,-1)
    chi = np.linspace(0.5/N, 1-0.5/N, num=N).reshape(-1,1)
    x_tnsr = chi*x_G_tnsr
    if not test_mode:
        x_shelf_tnsr = np.concatenate(x_shelf_list, axis=1)
        H_shelf_tnsr = np.concatenate(H_shelf_list, axis=1)

    if np.isnan(h_tnsr).any():
        converge = False
    else:
        converge = True

    converge = not np.isnan(h_tnsr).any()
    if not hide_output:
        print(f"Converged: {converge}")

    if test_mode:
        out = x_tnsr, h_tnsr, converge, x_G_tnsr
    
    else:
        out = x_tnsr, h_tnsr, x_shelf_tnsr, H_shelf_tnsr

    return out