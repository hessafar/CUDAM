import os
import numpy as np

# This reads the run parameters from an available input file
def read_parameter(st_line): 
    ch = ""
    value = ""
    pname_end = False
    pvalue_start = False
    f_allowed = [".", "e", "-", "+"]
    for char in st_line:
        ch += char
        if (char == "=" or char == " ") and not pname_end:
            param = ch
            param = param.replace("=", "")
            param = param.replace(" ", "")
            pname_end = True
            st_line = st_line.replace(ch, "")
            
        if (char.isdigit() or char in f_allowed) and pname_end:
            value += char
            pvalue_start = True
        if pvalue_start and (char == " "):
            break
    value = float(value)
    return param, value

result_path = "Results_datadriven/"
outfile = "output"
outfile_path = "output_path_long"
outfile_cooling = "output_cooling"
check_file = result_path + outfile
check_file_path = result_path + outfile_path
check_file_cooling = result_path + outfile_cooling
check_file_laser_distance = result_path + "laser_distance.h5"
check_file_meltpool = result_path + "meltpool_1.h5"
check_file =[check_file+".h5", check_file+".xdmf", check_file_path+".h5", check_file_cooling+".h5", 
            check_file_laser_distance, check_file_meltpool]

def delete_all_files():
    # input_file = "run_parameters_316L_varP_highRes.inp"
    # input_file = "run_parameters_316L.inp"
    # result_path = "Results/"
    

    try:
        # Check if the file exists
        for i in range(len(check_file)):
            if os.path.exists(check_file[i]):
                # Delete the file
                os.remove(check_file[i])
                print("The prevously available output file ({}) has been deleted!".format(check_file[i]))
    except Exception as er:
        print("An error occurred while trying to delete the file, " \
            "make sure the available files are closed in any application: {}".format(er))
        exit()

input_file = "run_parameters_316L_datadriven.inp"
param_dict = dict()
try:
    with open(input_file, "r") as file:
        # line = file.readline()
        for line in file:
            if line[0] != "#":
                param, value = read_parameter(line)
                param_dict.update({param: value})
    
except Exception as er:
    print("The input file doesn't exist! {}".format(er))
    exit()


path = np.array([1, 3, 2, 0])

rho = param_dict["rho"]  # Density, unit: kg/m3
cp = param_dict["cp"]  # Specific heat capacity, unit: J/(kg K)
k = param_dict["k"]  # Theramal conductivity, unit: W/(m K)
T_l = param_dict["T_l"]  # Liquidus temperature (K)
T_s = param_dict["T_s"]  # Solidus temperature (K)
H_l = param_dict["H_l"]  # Phase change enthalpy (J/kg)
T_inf = param_dict["T_inf"]  # Ambient temperature (K)
h_conv = param_dict["h_conv"]  # Convection heat transfer coefficint (W/(m2K))

length = param_dict["length"]  # Length in m - Corresponding to x-axis
width = param_dict["width"]  # Width in m - Corresponding to y-axis
height = param_dict["height"]  # Height in m - Corresponding to z-axis
line_length = param_dict["line_length"]  # Length of the laser scan

d_laser = param_dict["d_laser"]  # Diameter of the laser (m)
v_laser = param_dict["v_laser"]  # Velocity of the laser (m/s)
depth_laser = param_dict["depth_laser"]
P_laser = param_dict["P_laser"]  # Laser power (W) * alpha

nx = int(param_dict["nx"])  # Number of grid points in x direction
ny = int(param_dict["ny"])  # Number of grid points in y direction
nz = int(param_dict["nz"])  # Number of grid points in z direction
dt_coeff = param_dict["dt_coeff"]  # dt = dt_coeff * min(dx, dy, dz)  # Determinig the time step size
dt = param_dict["dt"]  # dt = dt_coeff * min(dx, dy, dz)  # Determinig the time step size

bl_x = int(param_dict["bl_x"])
bl_y = int(param_dict["bl_y"])
bl_z = int(param_dict["bl_z"])

f_write = int(param_dict["f_write"])  # Frequency of writing the output results in h5 file (in time steps)
f_mp = int(param_dict["f_meltpool"]) # Frequency of computing the melt pool dimensions (in time steps)

C_L = param_dict["C_L"]
C_Wi = param_dict["C_Wi"]
C_H = param_dict["C_H"]
C_t =param_dict["C_t"]
C_T = param_dict["C_T"]
C_v = C_L / C_t
C_alpha = C_L**2/C_t
C_M = rho * C_L**3
C_W = C_M * C_L**2 / C_t**3
C_k = C_W / (C_L * C_T)
C_rhocp = C_k / C_alpha
k_hat = k / C_k
rhocp_hat = rho * cp / C_rhocp


L = length / C_L
W = width / C_Wi
H = height / C_H

d_laser = d_laser / C_L  # Diameter of the laser (m)
r_laser = 0.5 * d_laser  # Raduis of the laser (m)
v_laser = v_laser / C_v  # Velocity of the laser (m/s)
depth_laser = depth_laser / C_L  # Set it to 2
P0_hat = P_laser / C_W

# alpha = 0.01  # Thermal diffusivity

T_0 = 300.0 / C_T # Initial temperature (K)

dx = L / (nx - 1)  # Grid spacing in x direction
dy = W / (ny - 1)  # Grid spacing in y direction
dz = H / (nz - 1)  # Grid spacing in z direction
# dt = dt_coeff * min(dx, dy, dz)  # Time step size

alpha = k / (rho * cp) # / C_alpha
time_final = L / v_laser
ts_final = int(time_final / dt)  # Number of time steps
n_section = path.shape[0]
ts_section = int(ts_final / n_section)
Q = 0  # Heat source term

T0 = T_0 * np.ones((nx * ny * nz))  # Initial temperature array
Tb = 300.0 / C_T  # Boundary temperature


# @cuda.jit
# def getInd(i, j, width):
#     return i * width + j

spacing_x = dx*C_L
spacing_y = dy*C_Wi
spacing_z = dz*C_H
origin_x, origin_y, origin_z = (0.0, 0.0, 0.0)
dim = (nx, ny, nz)
spacing = (spacing_x, spacing_y, spacing_z)
origin = (origin_x, origin_y, origin_z)