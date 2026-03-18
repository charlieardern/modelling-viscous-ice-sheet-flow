import numpy as np
from tqdm.auto import tqdm
from scipy.optimize import root_scalar
from scipy.linalg import solve_banded

class IceSheetSolver():

    def __init__(self, x_b, b, x_G_0, h_0, D, nu, L, num_microsteps, t_final, num_steps=100, rho=917, rho_w=1000, g=9.81):
        self.x_b = x_b
        self.b = b
        self.x_G_0 = x_G_0
        self.h_0 = h_0
        self.D = D
        self.nu = nu
        self.L = L
        self.num_microsteps = num_microsteps
        self.num_steps = num_steps
        self.t_final = t_final
        self.N = h_0.shape[0]
        self.rho = rho
        self.rho_w = rho_w
        self.g = g
        self.g_prime = self.g*(self.rho_w-self.rho)/self.rho_w
        self.C = (g/self.g_prime)**(1/6)
        self.a = (D*L/(2*self.C**4))**3*(g/(6*nu*L))
        self.B = (6*nu*self.a*L/g)**(1/3)
        

    def _K_plus(self, h, b_i_plus):
        h_plus = h[1:]
        h_i = h[:-1]
        b_i_plus_slice = b_i_plus[:-1]
        return (0.5*(h_i+h_plus)+b_i_plus_slice)**3
    
    def _K_minus(self, h, b_i_minus):
        h_i = h[1:]
        h_minus = h[:-1]
        b_i_minus_slice = b_i_minus[1:]
        return (0.5*(h_i+h_minus)+b_i_minus_slice)**3

    def _x_G_dot(self, h, x_G, h_G, b_G, db_G_dchi):
        dh_dchi = (8*h_G+h[-2]-9*h[-1])*self.N/3
        term_1 = 1.5*(self.rho_w*b_G/self.rho)**2*(1/x_G)*(dh_dchi**2-x_G**2)/((self.rho_w-self.rho)*db_G_dchi/self.rho-dh_dchi)
        term_2 = -(self.rho_w*b_G/self.rho)**2*dh_dchi/x_G
        return np.min([term_1, term_2])

    def _advective_term(self, h, x_G, h_G, dx_G_dt, chi):

        # Equation to be solved to find h_{-1}
        def f(h_neg): return (0.5*(h[0]+h_neg))**3*(h[0]-h_neg)*self.N+x_G

        if dx_G_dt > 0:
            h_minus = np.concatenate([np.zeros(1), h[:-1]], axis=0)
            h_minus_minus = np.concatenate([np.zeros(2), h[:-2]], axis=0)
            try:
                h_neg = root_scalar(f, bracket=[0, 5], method="brentq").root # h_{-1}
            except ValueError:
                h_neg = np.nan

            term_1 = 0.5*(3*h-4*h_minus+h_minus_minus)
            term_1[0] = 0.5*(h[1]-h_neg)
            term_1[1] = 0.5*(3*h[1]-4*h[0]+h_neg)
        
        else:
            h_plus = np.concatenate([h[1:], np.zeros(1)], axis=0)
            h_plus_plus = np.concatenate([h[2:], np.zeros(2)], axis=0)
            h_N = (8*h_G-6*h[-1]+h[-2])/3
            term_1 = 0.5*(-3*h+4*h_plus-h_plus_plus)
            term_1[-1] = 0.5*(h_N-h[-2])
            term_1[-2] = 0.5*(-3*h[-2]+4*h[-1]-h_N)
        
        return self.N*dx_G_dt*chi*term_1/x_G

    def _compute_shelf(self, t, num_microsteps, x_G_list, H_G_list, q_G_list, x_G_dot_list, t_G_list):
        x_list = []
        x_G_arr = 2*self.B*self.C**4*np.array(x_G_list)
        H_G_arr = np.array(H_G_list)*self.B*self.C
        q_G_arr = np.array(q_G_list)*self.B**3/2
        x_G_dot_arr = np.array(x_G_dot_list)*self.a*self.L/(self.B*self.C)
        t_G_arr = np.array(t_G_list)*2*self.B**2*self.C**5/(self.a*self.L)
        dt_scaled = self.delta_t*2*self.B**2*self.C**5/(self.a*self.L)
        t_scaled = t*2*self.B**2*self.C**5/(self.a*self.L)

        for iii in range(1+round((len(x_G_list))/num_microsteps)):
            idx = num_microsteps*iii
            term_1 = np.sum(dt_scaled*q_G_arr[idx:]/H_G_arr[idx:])
            term_2 = (self.g_prime/(8*self.nu))*np.sum(dt_scaled*(t_scaled-t_G_arr[idx:])*(q_G_arr[idx:]-H_G_arr[idx:]*x_G_dot_arr[idx:]))
            x_i = x_G_arr[idx]+term_1+term_2
            x_list.append(x_i)
        x_arr = np.array(x_list)
        H_arr = 8*self.nu*H_G_arr[::num_microsteps]/(self.g_prime*H_G_arr[::num_microsteps]*(t_scaled-t_G_arr[::num_microsteps])+8*self.nu)
        return x_arr/(2*self.B*self.C**4), H_arr/(self.B*self.C)

    def compute_solution(self, compute_shelf=True, hide_output=False):

        j = np.arange(0,self.N)
        chi_plus = (j+1)/self.N
        chi_minus = j/self.N
        chi = (j+0.5)/self.N
        self.delta_t = self.t_final/(self.num_steps*self.num_microsteps)
        self.delta_chi = 1/self.N

        x_G = self.x_G_0
        h = self.h_0

        b_G = np.interp(np.array(x_G), self.x_b, self.b)
        db_G_dchi = np.interp(np.array(x_G), self.x_b, np.gradient(self.b, self.x_b))
        h_G = (self.rho_w-self.rho)*b_G/self.rho
        H_G = h_G + b_G

        # Lists for final results
        h_list = []
        x_G_list = []
        x_shelf_list = []
        H_shelf_list = []

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
        x_G_dot_list.append(self._x_G_dot(h, x_G, h_G, b_G, db_G_dchi))
        t_G_list.append(0)
        H_G_list.append(H_G)
        q_G_list.append(-self.g/(3*self.nu)*H_G**3*((8*h_G+h[-2]-9*h[-1])*self.N/(3*x_G)))

        diverged = False

        for i in tqdm(range(self.num_steps), disable=hide_output):
            # Calculate shelf:
            t = i*self.num_microsteps*self.delta_t
            if compute_shelf:
                x_shelf, H_shelf = self._compute_shelf(t, self.num_microsteps, x_G_full_list, H_G_list, q_G_list, x_G_dot_list, t_G_list)
                x_shelf_list.append(np.pad(x_shelf, (0, 100-len(x_shelf)), mode='edge').reshape(-1,1))
                H_shelf_list.append(np.pad(H_shelf, (0, 100-len(H_shelf)), mode='edge').reshape(-1,1))

            # Calculate sheet:
            for ii in range(self.num_microsteps):
                b_G = np.interp(np.array(x_G), self.x_b, self.b)
                db_G_dchi = np.interp(np.array(x_G), self.x_b, np.gradient(self.b, self.x_b))
                
                h_G = (self.rho_w-self.rho)*b_G/self.rho
                
                b_i_minus = np.interp(chi_minus*x_G, self.x_b, self.b)
                b_i_plus = np.interp(chi_plus*x_G, self.x_b, self.b)
                

                dx_G_dt = self._x_G_dot(h, x_G, h_G, b_G, db_G_dchi)
                k_plus = self._K_plus(h, b_i_plus)
                k_minus = self._K_minus(h, b_i_minus)
                C_n = self.delta_t*self.N**2/(x_G**2)
                B_n = (self.rho_w*b_G/self.rho)**3/3

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

                M = lower + diagonal + upper

                const_term = np.zeros(self.N)
                const_term[0] = C_n*x_G/self.N
                const_term[-1] = 8*C_n*B_n*h_G

                adv = self._advective_term(h, x_G, h_G, dx_G_dt, chi)

                # faster solve:
                up = np.diag(M, k=1)
                di = np.diag(M, k=0)
                lo = np.diag(M, k=-1)
                ab = np.zeros((3, self.N))
                ab[0, 1:] = up
                ab[1, :] = di
                ab[2, :-1] = lo
                try:
                    h = solve_banded((1,1), ab, h+const_term+(adv+self.D)*self.delta_t)
                except ValueError:
                    h = h*np.nan
                
                x_G = x_G + dx_G_dt*self.delta_t
                x_G_full_list.append(x_G)
                h_full_list.append(h)
                x_G_dot_list.append(self._x_G_dot(h, x_G, h_G, b_G, db_G_dchi))
                H_G = h_G+b_G
                H_G_list.append(H_G)
                q_G_list.append(-self.g/(3*self.nu)*H_G**3*((8*h_G+h[-2]-9*h[-1])*self.N/(3*x_G)))
                t_G_list.append(i*self.num_microsteps*self.delta_t+(ii+1)*self.delta_t)

                if np.isnan(np.array(h)).any() and not diverged:
                    diverged = True
                    if not hide_output:
                        print("Solution has diverged.")
                    break

            h_list.append(h.reshape(-1,1))
            x_G_list.append(x_G)
            if diverged:
                break
        
        self.h_tnsr = np.concatenate(h_list, axis=1)
        self.x_G_tnsr = np.array(x_G_list).reshape(1,-1)
        self.x_tnsr = self.x_G_tnsr*chi.reshape(-1,1)

        if compute_shelf:
            self.x_shelf_tnsr = np.concatenate(x_shelf_list, axis=1)
            self.H_shelf_tnsr = np.concatenate(H_shelf_list, axis=1)
        else:
            self.x_shelf_tnsr = None
            self.H_shelf_tnsr = None
        
        converge = not np.isnan(self.h_tnsr).any()
        if not hide_output:
            print(f"Converged: {converge}")
        