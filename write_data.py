import h5py
import numpy as np
from globals import *


def write_vtk(filename, data): #, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):

    # Convert the NumPy array to a VTK array
    vtk_data_array = numpy_to_vtk(num_array=data.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

    # Define the scalar name
    vtk_data_array.SetName("Temperature")

    # Create a VTK image data object
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(data.shape)

    # Set the grid spacing (dx, dy, dz)
    # vtk_image.SetSpacing(dx, dy, dz)
    vtk_image.SetSpacing(dx, dy*C_Wi/C_L, dz*C_H/C_L)

    # Set the origin (optional)
    origin_x, origin_y, origin_z = 0.0, 0.0, 0.0
    vtk_image.SetOrigin(origin_x, origin_y, origin_z)

    # Attach the VTK data array to the image data object
    vtk_image.GetPointData().SetScalars(vtk_data_array)

    # Write the VTK image data to a .vtk file
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(filename)
    writer.SetInputData(vtk_image)
    writer.Write()
 

def write_hdf_3d_temporal(filename, parameters, data, time, data_write, first_write, ending):
    dim = data.shape
    time = time*dt
    file_h5 = filename + ".h5"
    file_xdmf = filename + ".xdmf"
    if data_write:
        with h5py.File(result_path + file_h5, "a") as h5file:

            
            # Create a group for time steps
            time_group = h5file.create_group("Timestep_{}".format(time))

            if first_write:
                param_group = h5file.create_group("SimulationParameters")
                key_l = list(parameters.keys())
                for i in range (len(key_l)):
                    param_group.attrs[key_l[i]] = parameters[key_l[i]]
                
                # Create a group for the structured grid
                grid_group = h5file.create_group("structuredGrid")
                
                # Add grid dimensions,origin, and spacing as attributes
                grid_group.attrs["dims"] = dim
                grid_group.attrs["orign"] = origin
                grid_group.attrs["spacing"] = spacing
                
                # Create datasets for each coordinate axis
                x = np.linspace(origin[0], origin[0] + (dim[0] - 1) * spacing_x, dim[0])
                y = np.linspace(origin[1], origin[1] + (dim[1] - 1) * spacing_y, dim[1])
                z = np.linspace(origin[2], origin[2] + (dim[2] - 1) * spacing_z, dim[2])
                
                grid_group.create_dataset("X", data=x)
                grid_group.create_dataset("Y", data=y)
                grid_group.create_dataset("Z", data=z)
        
            # Create a dataset for the temperature data
            temperature_dataset = time_group.create_dataset("Temperature".format(time), data=data)
            temperature_dataset.attrs["Unit"] = "Kelvin"  # Add unit attribute if needed
        
            
        xdmf_content = f"""
        <Grid Name="structuredGrid_t{time}" GridType="Uniform">
            <Time Value="{time}"/>
            <Topology TopologyType="3DRectMesh" Dimensions="{dim[2]} {dim[1]} {dim[0]}"/>
            <Geometry GeometryType="VxVyVz">
            <DataItem Format="HDF" Dimensions="{dim[0]}" NumberType="Float" Precision="4">
                {file_h5}:/structuredGrid/X
            </DataItem>
            <DataItem Format="HDF" Dimensions="{dim[1]}" NumberType="Float" Precision="4">
                {file_h5}:/structuredGrid/Y
            </DataItem>
            <DataItem Format="HDF" Dimensions="{dim[2]}" NumberType="Float" Precision="4">
                {file_h5}:/structuredGrid/Z
            </DataItem>
            </Geometry>
            <Attribute Name="Temperature" AttributeType="Scalar" Center="Node">
            <DataItem Dimensions="{dim[2]} {dim[1]} {dim[0]}" NumberType="Float" Precision="4" Format="HDF">
                {file_h5}:/Timestep_{time}/Temperature
            </DataItem>
            </Attribute>
        </Grid>
        """
    
    if first_write:
        xdmf_content0 = f"""<?xml version="1.0" ?>
        <Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
        <Domain>
            <Grid GridType="Collection" CollectionType="Temporal">
        """
        xdmf_content = xdmf_content0 + xdmf_content
    
    if ending:
        xdmf_content = """
        </Grid>
        </Domain>
        </Xdmf>
        """
    
    with open(result_path + file_xdmf, "a") as xdmf_file:
        xdmf_file.write(xdmf_content)
        
def write_hdf_3d_plane(filename, parameters, data, time, data_write, first_write, ending):
    dim = data.shape
    file_h5 = filename + ".h5"
    file_xdmf = filename + ".xdmf"
    dim = (nx, ny, 1)
    with h5py.File(result_path + file_h5, "a") as h5file:

        
        # Create a group for time steps
        time_group = h5file.create_group("Timestep_{}".format(time))

        if first_write:
            param_group = h5file.create_group("SimulationParameters")
            key_l = list(parameters.keys())
            for i in range (len(key_l)):
                param_group.attrs[key_l[i]] = parameters[key_l[i]]
            
            # Create a group for the structured grid
            grid_group = h5file.create_group("structuredGrid")
            
            # Add grid dimensions,origin, and spacing as attributes
            grid_group.attrs["dims"] = dim
            grid_group.attrs["orign"] = origin
            grid_group.attrs["spacing"] = (spacing_x, spacing_y, 1.0)
            
            # Create datasets for each coordinate axis
            x = np.linspace(origin[0], origin[0] + (dim[0] - 1) * spacing_x, dim[0])
            y = np.linspace(origin[1], origin[1] + (dim[1] - 1) * spacing_y, dim[1])
            z = np.linspace(origin[2], origin[2] + (dim[2] - 1) * spacing_z, dim[2])
            
            grid_group.create_dataset("X", data=x)
            grid_group.create_dataset("Y", data=y)
            grid_group.create_dataset("Z", data=z)
    
        # Create a dataset for the temperature data
        temperature_dataset = time_group.create_dataset("Temperature".format(time), data=data.T)
        temperature_dataset.attrs["Unit"] = "Kelvin"  # Add unit attribute if needed
        
def write_hdf_3d_plane_path(filename, parameters, data, path_number, time, plane):
    dim = data.shape
    file_h5 = filename + ".h5"
    file_xdmf = filename + ".xdmf"
    dim = (nx, ny, nz)
    if type(time) == int: time = time*dt
    with h5py.File(result_path + file_h5, "a") as h5file:

        path_group_name = f"path{path_number}"
        grid_group_name = "structuredGrid"
        param_group_name = "simulationParameters"
        time_group_name = f"{time:0.6f}" if type(time) is not str else time
        
        if path_group_name in h5file:
            path_group = h5file[path_group_name]
        else:
            path_group = h5file.create_group(path_group_name)
        
        if time_group_name in path_group:
            time_group = path_group[time_group_name]
        else:
            time_group = path_group.create_group(time_group_name)

        if grid_group_name not in h5file:
            param_group = h5file.create_group(param_group_name)
            key_l = list(parameters.keys())
            for i in range (len(key_l)):
                param_group.attrs[key_l[i]] = parameters[key_l[i]]
        
        if grid_group_name not in h5file:
            grid_group = h5file.create_group(grid_group_name)
            # Add grid dimensions,origin, and spacing as attributes
            grid_group.attrs["dims"] = dim
            grid_group.attrs["orign"] = origin
            grid_group.attrs["spacing"] = spacing

            # Create datasets for each coordinate axis
            x = np.linspace(origin[0], origin[0] + (dim[0] - 1) * spacing_x, dim[0])
            y = np.linspace(origin[1], origin[1] + (dim[1] - 1) * spacing_y, dim[1])
            z = np.linspace(origin[2], origin[2] + (dim[2] - 1) * spacing_z, dim[2])
            
            grid_group.create_dataset("X", data=x)
            grid_group.create_dataset("Y", data=y)
            grid_group.create_dataset("Z", data=z)

        # Create a dataset for the temperature data
        temperature_dataset = time_group.create_dataset(f"Temperature_{plane}", data=data.T)
        temperature_dataset.attrs["Unit"] = "Kelvin"  # Add unit attribute if needed

def write_hdf_meltpool_dim(filename, data, path_number):
    with h5py.File(filename, "a") as h5file:

        path_group_name = f"path{path_number}"
        time_group_name = "times"
        
        if time_group_name not in h5file:
            time_group = h5file.create_group(time_group_name)
            time_group.create_dataset("time", data=data[:,0])
        

        if path_group_name not in h5file:
            path_group = h5file.create_group(path_group_name)
            path_group.create_dataset("length", data=data[:,1])
            path_group.create_dataset("width", data=data[:,2])
            path_group.create_dataset("height", data=data[:,3])
            path_group.create_dataset("dheight", data=data[:,4])
            path_group.create_dataset("x_loc", data=data[:,5])
   

def compute_melt_pool_dimensions(data_xy, data_xz, T_melt, dx, dy, dz):
    # Create a mask where the temperature is above or equal to the melting temperature
    
    i_max = data_xy.shape[0]
    j_max = data_xy.shape[1]
    k_max = data_xz.shape[1]
    # dx = x_coord[1] - x_coord[0]
    # dy = y_coord[1] - y_coord[0]
    length = 0.0
    width = 0.0
    depth = 0.0
    low_end = 0.0
    high_end = 0.0

    for j in range(j_max):
        for i in range(i_max - 1):
            if data_xy[i, j] < T_melt and data_xy[i+1, j] > T_melt:
                low_end = i*dx + dx*(T_melt - data_xy[i, j]) / (data_xy[i+1, j] - data_xy[i, j])
            if data_xy[i, j] > T_melt and data_xy[i+1, j] < T_melt:
                high_end = i*dx + dx*(T_melt - data_xy[i, j]) / (data_xy[i+1, j] - data_xy[i, j])
            if (high_end - low_end) > length: length = high_end - low_end

    low_end = 0.0
    high_end = 0.0
    for i in range(i_max):
        for j in range(j_max - 1):
            if data_xy[i, j] < T_melt and data_xy[i, j+1] > T_melt:
                low_end = j*dy + dy*(T_melt - data_xy[i, j]) / (data_xy[i, j+1] - data_xy[i, j])
            if data_xy[i, j] > T_melt and data_xy[i, j+1] < T_melt:
                high_end = j*dy + dy*(T_melt - data_xy[i, j]) / (data_xy[i, j+1] - data_xy[i, j])
            if (high_end - low_end) > width: 
                width = high_end - low_end

    
        #return length*1000, width*1000
    
    x_loc = 0
    depth_temp = 0
    for i in range(i_max):
        for k in range(k_max - 1):
            if data_xz[i, k] < T_melt and data_xz[i, k+1] > T_melt:
                depth_temp = height - (k*dz + dz*(T_melt -data_xz[i,k]) / (data_xz[i,k+1] - data_xz[i,k]))
                # depth_temp =(j*dy + dy*(T_melt -data[i,j]) / (data[i,j+1] - data[i,j]))
            if depth_temp > depth:
                depth = depth_temp
                x_loc = i*dx

        
    return length*1000, width*1000, depth*1000, x_loc