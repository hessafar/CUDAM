# Import Numba and its CUDA module
import numpy as np
import numba
from numba.cuda.cudadrv.driver import CudaAPIError
from numba import cuda
import math
import time
from matplotlib import pyplot as plt
# import vtk
# from vtk.util.numpy_support import numpy_to_vtk
import h5py
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.interpolate import interp1d
from diffusion_pde import *
from write_data import *
from globals import *
from scipy.optimize import differential_evolution
import gc


print("dx = {}, dy = {}, dz = {}, dt = {}, alpha = {}".format(dx, dy, dz, dt, alpha))

counter = 0
def run_case(dP1, dP2, t0, dt1, dt2, dt3):
    from diffusion_pde import update_T, apply_BC
    delete_all_files()
    # data = np.load("square_wave_40samples.npz")
    # data = np.load("sine_samples_sobol_test.npz")
    # x_test_loaded = data["x_value"]
    # y_samples_loaded = data["y_value"]
    # data = np.load("modulations_40samples.npz")
    

    # x_test_loaded = data["t"]  # Loaded x values
    # y_samples_loaded = data["p_sample"]
    global counter
    counter += 1
    print(f"{counter=}")

    rng = np.random.default_rng(seed=42)

    # num_sc = len(p_overshoot)
    num_sc = len(dP1)
    reach_destination = 0
    hatch_spacing = num_sc*[0.25*d_laser] # for varP
    # hatch_spacing = rng.uniform(0.1e-3, 0.3e-3, size=num_sc)
    # hatch_spacing = 0.8*d_laser
    sc1 = np.array([0])
    sc2 = np.array([0])
    sc3 = np.array([0])
    sc4 = np.array([0])
    sc5 = np.array([0])
    sc6 = np.array([0])
    sc_test = np.array([0])

    scenarios = np.array(num_sc*[sc1])
    # scenarios = np.array([sc_test])
    # scenarios = generate_path(n_section=3, n_path=6)
    print(f"Scenario shape: {scenarios.shape}")

    # line_order = 
    jump_direction = (0, 1)
    laser_direction = (1, 0)
    X_start = (0.5e-3, 0.5e-3)
    # line_length = 1e-3 #0.999e-3
    line_process_time = (line_length) / v_laser
    line_total_ts = int(line_process_time / dt)
    t_n = np.linspace(0, line_process_time, line_total_ts).reshape(-1, 1)
    # p_samples = gp.sample_y(t_n, n_samples=1, random_state=20) + P_laser
    # print("p_samples shape", p_samples.shape)
    print(f"process time: {line_process_time}, final time step: {line_total_ts}")
    x_laser = X_start[0]
    y_laser = X_start[1]
    # while not reach_destination:
    with_cooling = False

    # Copy the initial temperature array to the device memory
    # T_old = cuda.to_device(T0.copy())
    T_old = cuda.to_device(300.0 * np.ones((nx * ny * nz)))
    molten = cuda.to_device(np.zeros((nx * ny * nz), dtype=bool))
    # Allocate an empty array for the new temperature on the device memory
    T_new = cuda.device_array_like(T_old)

    # Define the number of threads per block and the number of blocks per grid
    threads_per_block = (6, 6, 6)

    blocks_per_grid_x = int(np.ceil(nx / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(ny / threads_per_block[1]))
    blocks_per_grid_z = int(np.ceil(nz / threads_per_block[2]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)
    print(blocks_per_grid)

    # Loop over the time steps
    start = time.perf_counter()
    t = 0.0
    ts = 0
    first_write = 1


    simulation_time = line_total_ts
    # simulation_time = 3000
    x0, y0 = 0.0, 0.0

    paraview_output = False
    for path in range(scenarios.shape[0]):
        print(f"running scenario no. {path}")
        laser_distance, laser_time = [], []
        mp_dims = []
        ts = 0
        reset = 1
        cooling = False
        
        # selected_sine_function = y_samples_loaded[path, :]
        for line_order in range(scenarios.shape[1]):
            x_laser = X_start[0] + scenarios[path, line_order]*hatch_spacing[path]*jump_direction[0]
            y_laser = X_start[1] + scenarios[path, line_order]*hatch_spacing[path]*jump_direction[1]
            for path_time in range(simulation_time):
                # t += dt
                if ts > line_total_ts:
                    laser_time.append(ts*dt)
                    distance = math.sqrt((x_laser - x0)**2 + (y_laser - y0)**2)
                    laser_distance.append(distance)
                ts += 1
                # if ts % 100 == 0: print(ts, x_laser, y_laser)
                # function_interpolator = interp1d(x_test_loaded.ravel(), selected_sine_function, kind='cubic', fill_value="extrapolate")
                #p_t = function_interpolator(np.array([path_time*dt]))

                #p_t = p_samples[path_time, path]
                # p_t = [P_laser+jump]
                
                real_time = path_time*dt
                t_r = path_time*dt
                
                # jump = 0 if real_time < t_0[path] or real_time > (t_0[path]+duration[path]) else p_overshoot[path]
                
                # f_t = 0.0
                # if real_time > t_0[path] and real_time  < t_0[path] + duration[path]:
                #     f_t = real_time - t_0[path]
                # if real_time >= t_0[path] + duration[path]:
                #     f_t = duration[path]
                # jump = p_overshoot[path] * f_t/duration[path]

                jump = 0.0
                if t0[path] < t_r < t0[path] + dt1[path]:
                    jump = dP1[path] * (t_r - t0[path]) / dt1[path]
                elif t0[path] + dt1[path] <= t_r <= t0[path] + dt1[path] + dt2[path]:
                    jump = dP1[path]
                elif t0[path] + dt1[path] + dt2[path] < t_r < t0[path] + dt1[path] + dt2[path] + dt3[path]:
                    jump = dP1[path] + dP2[path] * (t_r - (t0[path] + dt1[path] + dt2[path])) / dt3[path]
                elif t_r >= t0[path] + dt1[path] + dt2[path] + dt3[path]:
                    jump = dP1[path] + dP2[path]

                p_t = P_laser + jump
                

                # x_laser += v_laser * dt
                # Call the kernel function with the grid and block configuration
                try:
                    update_T[blocks_per_grid, threads_per_block](T_new, T_old, molten, dt, dx, dy, x_laser, y_laser, reset, cooling, p_t)
                    cuda.synchronize()
                except CudaAPIError as e:
                    print(f"KERNEL ERROR at time step {ts}: {e}")
                    raise
                try:
                    apply_BC[blocks_per_grid, threads_per_block](T_new)
                    cuda.synchronize()
                except CudaAPIError as e:
                    print(f"KERNEL ERROR at time step {ts}: {e}")
                    raise
                reset = 0
                # Swap the old and new temperature arrays
                T_old, T_new = T_new, T_old
                
                
                x_laser += v_laser * laser_direction[0] * dt
                y_laser += v_laser * laser_direction[1] * dt
                

                if ts % f_write == 0:
                    T_write = T_old.copy_to_host()
                    
                    #for viewing in Paraview
                    if paraview_output:
                        T_write = C_T*T_write.reshape((nx, ny, nz))
                        write_hdf_3d_temporal(filename=outfile, parameters=param_dict, data=T_write, time=ts, data_write=1, first_write=first_write, ending=0)
                        first_write = 0
                    else:
                    # for viewing in matplotlib
                        T_write = C_T*T_write.reshape((nz, ny, nx))
                        T_xy = T_write[-1,:,:]
                        T_xy = T_xy.reshape((T_xy.shape[0], T_xy.shape[1], 1))
                        y_ind = int(y_laser/dy)
                        T_xz = T_write[:,y_ind,:]
                        T_xz = T_xz.reshape((T_xz.shape[0], T_xz.shape[1], 1))
                        write_hdf_3d_plane_path(filename=outfile_path, parameters=param_dict, data=T_xy, path_number=path+1, time=ts, plane="xy")
                        write_hdf_3d_plane_path(filename=outfile_path, parameters=param_dict, data=T_xz, path_number=path+1, time=ts, plane="xz")
                        first_write = 0
                
                if ts % f_mp == 0:# and ts*dt >= 0.0015:
                    T_write = T_old.copy_to_host()
                    T_write = C_T*T_write.reshape((nz, ny, nx))
                    T_xy = T_write[-1,:,:]
                    T_xy = T_xy.reshape((T_xy.shape[0], T_xy.shape[1])).T
                    y_ind = int(y_laser/dy)
                    T_xz = T_write[:,y_ind,:]
                    T_xz = T_xz.reshape((T_xz.shape[0], T_xz.shape[1])).T
                    # print(T_xy.shape, T_xz.shape)
                    mp_l, mp_w, mp_d, x_loc = compute_melt_pool_dimensions(data_xy=T_xy, data_xz=T_xz,
                                                                    T_melt=T_s, dx=dx, dy=dy, dz=dz)
                    mp_dims.append([ts*dt, mp_l, mp_w, mp_d, 0.0, x_loc])
                    # print(mp_l, mp_w, mp_d)
        mp_dims = np.array(mp_dims)
        for ii in range(1, mp_dims.shape[0]-1):
            mp_dims[ii, 4] = mp_dims[ii+1, 3] - mp_dims[ii-1, 3]
        mp_dims[0, 4] = mp_dims[1, 4]
        mp_dims[-1, 4] = mp_dims[-2, 4]
        # write_hdf_meltpool_dim(filename=melt_pool_file, data=mp_dims, path_number=path+1)

        # T_write = T_old.copy_to_host()
        # T_write = C_T*T_write.reshape((nz, ny, nx))
        # T_xy = T_write[-1,:,:]
        # T_xy = T_xy.reshape((T_xy.shape[0], T_xy.shape[1], 1))
        # write_hdf_3d_plane_path(filename="laser_distance", parameters=param_dict, data=T_xy, 
        #                         path_number=path+1, time="end", plane="xy")

        # with h5py.File(result_path+"laser_distance.h5", 'a') as f:
        #     dataset_name = f"path_{path+1}"
        #     dataset_time = "time"
        #     f.create_dataset(dataset_name, data=laser_distance)
        #     if dataset_time not in f:
        #         f.create_dataset(dataset_time, data=laser_time)

        #### cooling phase ####
        if with_cooling:
            cooling = True
            ts = 0
            T_write = T_old.copy_to_host()
            # T_write = C_T*T_write.reshape((nx, ny, nz))
            T_write = C_T*T_write.reshape((nz, ny, nx))
            T_xy = T_write[-1,:,:]
            T_xy = T_xy.reshape((T_xy.shape[0], T_xy.shape[1], 1))
            y_ind = int(y_laser/dy)
            T_xz = T_write[:,y_ind,:]
            T_xz = T_xz.reshape((T_xz.shape[0], T_xz.shape[1], 1))
            # write_hdf_3d_plane(filename=outfile, parameters=param_dict, data=T_write, time=ts, data_write=1, first_write=first_write, ending=0)
            write_hdf_3d_plane_path(filename=outfile_cooling, parameters=param_dict, data=T_xy, path_number=path+1, time=ts, plane="xy")
            write_hdf_3d_plane_path(filename=outfile_cooling, parameters=param_dict, data=T_xz, path_number=path+1, time=ts, plane="xz")
            for _ in range(20001):
                ts += 1
                if ts % 100 == 0: print(ts, dt, x_laser, y_laser)
                update_T[blocks_per_grid, threads_per_block](T_new, T_old, dt, dx, dy, x_laser, y_laser, reset, cooling)
                cuda.synchronize()
                apply_BC[blocks_per_grid, threads_per_block](T_new)
                cuda.synchronize()
                # Swap the old and new temperature arrays
                T_old, T_new = T_new, T_old
                if ts % 2000 == 0: 
                        T_write = T_old.copy_to_host()
                        # T_write = C_T*T_write.reshape((nx, ny, nz))
                        T_write = C_T*T_write.reshape((nz, ny, nx))
                        T_xy = T_write[-1,:,:]
                        T_xy = T_xy.reshape((T_xy.shape[0], T_xy.shape[1], 1))
                        y_ind = int(y_laser/dy)
                        T_xz = T_write[:,y_ind,:]
                        T_xz = T_xz.reshape((T_xz.shape[0], T_xz.shape[1], 1))
                        # write_hdf_3d_plane(filename=outfile, parameters=param_dict, data=T_write, time=ts, data_write=1, first_write=first_write, ending=0)
                        write_hdf_3d_plane_path(filename=outfile_cooling, parameters=param_dict, data=T_xy, path_number=path+1, time=ts, plane="xy")
                        write_hdf_3d_plane_path(filename=outfile_cooling, parameters=param_dict, data=T_xz, path_number=path+1, time=ts, plane="xz")
                        first_write = 0

            
    end = time.perf_counter()
    print(f"The time is {end - start} seconds.")
    # Copy the final temperature array from the device to the host memory
    T_final = T_old.copy_to_host()

    T_output = C_T*T_final.reshape((nx, ny, nz))
    # print(T_output.shape) 

    # write_hdf_3d_temporal(filename=outfile, parameters=param_dict, data=T_output, time=ts, data_write=0, first_write=0, ending=True)

    # T_old = None
    # T_new = None
    # molten = None
    del T_old, T_new, molten
    cuda.current_context().deallocations.clear()
    gc.collect()
    cuda.synchronize()


    # laser_distance = np.array(laser_distance)
    # laser_time = np.array(laser_time)
    # print(laser_distance, laser_distance.shape)


    mp_dims = np.array(mp_dims)
    # plt.plot(mp_dims[:,0]*dt, mp_dims[:,1])
    # plt.plot(mp_dims[:,0]*dt, mp_dims[:,2])
    # plt.plot(mp_dims[:,0]*dt, mp_dims[:,3])
    # plt.savefig(result_path+"mp_history.png")
    # plt.show()
    print("Finished run_case")
    return mp_dims[:,2], mp_dims[:,3], mp_dims[:,5]


def objective(params):
        # a, b, c = params
        a, b, c, d, e, f = params
        width, _, _ = run_case(dP1=[a], dP2=[b], t0=[c], dt1=[d], dt2=[e], dt3=[f])
        # depth_profile = abs(depth_profile)
        # return np.var(depth_profile)
        return np.mean((width - width[0])**2)

X_low =  np.array([20.0, -25.0, 0.0025, 2e-6, 2e-6, 2e-6])
X_high =  np.array([40.0, -10.0, 0.0027, 5e-4, 5e-4, 5e-4])

bounds = [(X_low[0], X_high[0]), (X_low[1], X_high[1]), (X_low[2], X_high[2]),
          (X_low[3], X_high[3]), (X_low[4], X_high[4]), (X_low[5], X_high[5])]  # adjust as needed


result = differential_evolution(objective, bounds, workers=1)
print(f"dP1 = [{result.x[0]}]")
print(f"dP2 = [{result.x[1]}]")
print(f"t0 = [{result.x[2]}]")
print(f"dt1 = [{result.x[3]}]")
print(f"dt2 = [{result.x[4]}]")
print(f"dt3 = [{result.x[5]}]")
