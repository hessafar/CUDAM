import numba
from numba import cuda
import numpy as np
import math
from globals import *
# import mat_properties as prop

# CUDA device function that updates the temperature
@cuda.jit  # (device=True)
def update_T(T_new, T_old, molten, dt, dx, dy, x_laser, y_laser, reset, cooling, p_t):
    # Get the thread indices
    i, j, k1 = cuda.grid(3)
    # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

            
    # Check if the thread is within the grid bounds
    if i < (nx - 1) and j < (ny - 1) and k1 < (nz - 1):

        k_g = 0.017
        phi = 0.50
        
        
        # With the following indexing the 3d array of T has the form of T[z, y, x] or T[k, j, i]
        index = k1*nx*ny + j * nx + i
        im1jk = k1*nx*ny + j * nx + (i - 1)
        ip1jk = k1*nx*ny + j * nx + (i + 1)
        ijm1k = k1*nx*ny + (j - 1) * nx + i
        ijp1k = k1*nx*ny + (j + 1) * nx + i
        ijkm1 = (k1 - 1)*nx*ny + j * nx + i
        ijkp1 = (k1 + 1)*nx*ny + j * nx + i

        # x_c = c_laser
        y_c = 0.5 * W
        coeff = 6 * math.sqrt(3) / (math.pi * math.sqrt(math.pi) * r_laser ** 3)
        # P = coeff * P0_hat * math.exp(-3*((i*dx - c_laser) ** 2 + (j*dy - y_c) ** 2) / r_laser ** 2) * math.exp(
        #     -3*(k*dz - H) ** 2 / depth_laser ** 2)
        # coeff = 2.0 / (dep_laser * math.pi  * a_laser * b_laser)
        # P = coeff * P0_hat * math.exp(-2*(((i*dx - c_laser)/a_laser) ** 2 + ((j*dy - y_c)/ b_laser) ** 2 )) * math.exp(
        #     -1*(k1*dz - H) ** 2 / dep_laser ** 2)
        P = 0.0
        if not cooling: 
            P = coeff * p_t * math.exp(-3*(((i*dx - x_laser)/r_laser) ** 2 + ((j*dy - y_laser)/ r_laser) ** 2 )) * math.exp(
            -3*(k1*dz - H) ** 2 / depth_laser ** 2)
        
        if reset:
            Tijk   = T_0
            Tim1jk = T_0
            Tip1jk = T_0
            Tijm1k = T_0
            Tijp1k = T_0
            Tijkm1 = T_0
            Tijkp1 = T_0
        else:
            Tijk   = T_old[index]
            Tim1jk = T_old[im1jk]
            Tip1jk = T_old[ip1jk]
            Tijm1k = T_old[ijm1k]
            Tijp1k = T_old[ijp1k]
            Tijkm1 = T_old[ijkm1]
            Tijkp1 = T_old[ijkp1]
        
        if molten[index] == 0 and Tijk > T_s: molten[index] = 1
        

        def compute_rho(T):
            rho_s = 8256.42 - 0.2019*T - 1.3379e-4*T*T
            if molten[index] == 0:
                val = (1-phi)*rho_s
            else:
                val = rho_s
            return val

        x1 = 1.5e-3
        L = length #1e-3
        beta = 100.0
        def compute_cond(T, ind_i):
            k_s_1 = (5.9224 + 0.0165*T)
            k_s_2 = 2*k_s_1
            # beta = k_g/k_s
            if molten[index] == 0:
                # val = k_g * (2*k_s+k_g+2*phi*(k_s-k_g)) / (2*k_s+k_g-phi*(k_s-k_g))
                val1 = (k_g + (k_s_1-k_g)*(1-phi)**2)/4.0
                val2 = (k_g + (k_s_2-k_g)*(1-phi)**2)/4.0
                # return k_g*(1-phi)**2 + k_s*(phi/(1-phi))*(beta+phi*(1-beta))/(beta-phi*(1-beta))
            else:
                val1 = k_s_1
                val2 = k_s_2
            # val1 = k_s_1
            # val2 = k_s_2
            val = val1 + (0.5*(np.tanh(beta*(ind_i*dx-x1)/L)) + 0.5)*(val2-val1)
            return val

        def compute_cp(T):
            cp_a = 1.9327e-10*T**4 - 7.9999e-07*T**3 + 1.1407e-03*T**2 - 4.4890e-01*T + 1.0575e+03
            if T < 800:
                val = (0.362 + 2.118e-4*T)*1000
            elif T <= 900:
                val = (-0.946 + 0.295e-2*T - 1.379e-6*T*T)*1000
            else:
                val = (0.639 - 3.355e-6*T)*1000
            if molten[index] == 0:
                res = (1-phi) * val + phi*cp_a
            else:
                res = val
            return res
        
        rho1 = compute_rho(Tijk)
        c_p = compute_cp(Tijk)
        # print(f"{rho1=}")
        
        x1 = 1.8e-3
        x2 = 2.3e-3
        L = length #1e-3
        beta = 100.0
        def cond_H(T, ind_i):
            # val1 = (234.887443886408 + 0.018367198 * (T-273) + 225.10088 * math.tanh(0.0118783 * ((T-273) - 1543.815)))
            val1 = k #(234.887443886408 + 0.018367198 * (T-273) + 225.10088 * math.tanh(0.0118783 * ((T-273) - 1543.815)))
            val2 = 1.5*val1
            # val = val1 + 0.5*(val2-val1) * (math.tanh(beta*(ind_i*dx-x1)/L) + math.tanh(beta*(x2-ind_i*dx)/L))
            val = val1 + 0.5*(val2-val1) * (math.tanh(beta*(ind_i*dx-x1)/L))
            # if 0.8e-3 < i*dx < 1.1e-3: 
            #     return 0.5*val
            # else:
            #     return val
            # return k
            return val1

        def thermal_conductivity(x, x1, x2, k1, k2, L, alpha=20):
            return k1 + 0.5*(k2-k1) * (np.tanh(alpha*(x-x1)/L) + np.tanh(alpha*(x2-x)/L))
        
        # c_p = (446.337248 + 0.14180844 * (Tijk-273) - 61.431671211432 * math.exp(
        #             -0.00031858431233904*((Tijk-273)-525)**2) + 1054.9650568*math.exp(-0.00006287810196136*((Tijk-273)-1545)**2))
        
        # TT = C_T*Tijk + 300.0
        # kij = (11.82 + 0.0106*(TT)) / C_k
        # kim1jk = (11.82 + 0.0106*(Tim1jk*C_T + 300.0)) / C_k
        # kip1jk = (11.82 + 0.0106*(Tip1jk*C_T + 300.0)) / C_k
        # kijm1k = (11.82 + 0.0106*(Tijm1k*C_T + 300.0)) / C_k
        # kijp1k = (11.82 + 0.0106*(Tijp1k*C_T + 300.0)) / C_k
        # kijkm1 = (11.82 + 0.0106*(Tijkm1*C_T + 300.0)) / C_k
        # kijkp1 = (11.82 + 0.0106*(Tijkp1*C_T + 300.0)) / C_k
        # c_p = 330.9 + 0.563*TT  - 4.015e-4*(TT)**2 + 9.465e-8*(TT)**3
        # c_p +=  0.5*(H_l / (T_l - T_s)) * (math.exp(-(((TT - 1690.5)/(T_l - T_s))**2)))
        if T_s < Tijk < T_l:
            c_p = c_p + H_l / (T_l - T_s)

        
        kijk = compute_cond(Tijk, i)
        kim1jk = compute_cond(Tim1jk, i-1)
        kip1jk = compute_cond(Tip1jk, i+1)
        kijm1k = compute_cond(Tijm1k, i)
        kijp1k = compute_cond(Tijp1k, i)
        kijkm1 = compute_cond(Tijkm1, i)
        kijkp1 = compute_cond(Tijkp1, i)

        # kijk = compute_cond(Tijk)
        # kim1jk = compute_cond(Tim1jk)
        # kip1jk = compute_cond(Tip1jk)
        # kijm1k = compute_cond(Tijm1k)
        # kijp1k = compute_cond(Tijp1k)
        # kijkm1 = compute_cond(Tijkm1)
        # kijkp1 = compute_cond(Tijkp1)


        
        
        # rhocp_hat = rho*c_p / C_rhocp
        # dx_hat = (C_L/C_L)*dx
        # dy_hat = (C_Wi/C_L)*dy
        # dz_hat = (C_H/C_L)*dz

        # Constant properties
        # alpha1 = k/(rho*c_p)
        # T_new[index] = Tijk + dt * (alpha1 * ((Tip1jk - 2 * Tijk + Tim1jk) / dx ** 2 + (Tijp1k - 2 * Tijk + Tijm1k) / dy ** 2 + 
        #                                       (Tijkp1 - 2 * Tijk + Tijkm1) / dz ** 2) + P/(rho*c_p))
        
        
        # Variable properties
        rhs = (kijk*((Tip1jk - 2 * Tijk + Tim1jk) / dx ** 2 + (Tijp1k - 2 * Tijk + Tijm1k) / dy ** 2 + 
                   (Tijkp1 - 2 * Tijk + Tijkm1) / dz ** 2) + 0.25*(kip1jk - kim1jk)*(Tip1jk - Tim1jk) / dx**2 + 
              0.25*(kijp1k - kijm1k)*(Tijp1k - Tijm1k)/dy**2 + 0.25*(kijkp1 - kijkm1)*(Tijkp1 - Tijkm1)/dz**2)
            
        T_new[index] = Tijk + dt * (rhs + P) / (rho1*c_p)

@cuda.jit  # (device=True)
def apply_BC(T_new):
    # Get the thread indices
    i, j, k1 = cuda.grid(3)
    # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    if i < (nx - 1) and j < (ny - 1) and k1 < (nz - 1):
        index = k1*nx*ny + j * nx + i
        im1jk = k1*nx*ny + j * nx + (i - 1)
        ip1jk = k1*nx*ny + j * nx + (i + 1)
        ijm1k = k1*nx*ny + (j - 1) * nx + i
        ijp1k = k1*nx*ny + (j + 1) * nx + i
        ijkm1 = (k1 - 1)*nx*ny + j * nx + i
        ijkp1 = (k1 + 1)*nx*ny + j * nx + i

        a_laser = r_laser
        b_laser = r_laser / (C_Wi/C_L)
        y_c = 0.5 * W
        c1 = 1.0 #h_conv * dx / k

        if i == 0:
            T_new[index] = T_new[ip1jk]
        if i == (nx - 1):
            T_new[index] = T_new[im1jk]
        if j == (ny - 1):
            T_new[index] = T_new[ijm1k]
        if j == 0:
            T_new[index] = T_new[ijp1k]
        if k1 == 0:
            T_new[index] = Tb #T_new[ijkp1]
        if k1 == (nz - 1):
            # kijkm1 = (11.82 + 0.0106*T_new[ijkm1]*C_T) / C_k
            # q_l = P0_hat / (math.pi * r_laser ** 2) * math.exp(-2*(((i*dx - c_laser)/a_laser) ** 2 + ((j*dy - y_c)/ b_laser) ** 2 ))
            # # cq = q_l * C_H/C_L*dz / k_hat
            # cq = q_l * C_H/C_L*dz / kijkm1
            # T_new[index] = (c1 * T_inf*0 + cq + T_new[ijkm1])
            T_new[index] = T_new[ijkm1]