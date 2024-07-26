import os
import math

import numpy as np
import trimesh
from pysdf import SDF

from arg_parser import parse_args

# example usage: python hexagonal_sphere_packing.py --input ./bunny.obj --output ./out --resolution 25 --shadertoy-compatible --output-mesh --output-pc
if __name__ == '__main__':
    args = parse_args()
    resolution = args.resolution

    mesh = trimesh.load_mesh(args.input, process=False)

    # Mesh normalization to -0.9 to 0.9
    # Calculate bounding box
    box_size = 1.8
    bb_min = mesh.vertices.min(axis=0)
    bb_max = mesh.vertices.max(axis=0)
    center = (bb_max + bb_min) / 2.0

    # Translate vertices to the center
    new_V = mesh.vertices - center

    # Find the largest dimension of the bounding box
    len_box = max(bb_max - bb_min)

    # Scale model
    ratio = box_size / len_box
    new_V = ratio * new_V

    mesh.vertices = new_V

    # output the scaled mesh
    # mesh.export('scaled_mesh.ply')

    # create a SDF
    f = SDF(mesh.vertices, mesh.faces)

    # we want #resolution spheres along the largest dimension, which is 1.8 with scaling
    # but we use 2.0 to make sure the sphere is fully inside the bounding box
    sphere_radius = 2.0 / resolution / 2.0

    # x,y is each layer of the hexagonal packing
    # z is the layer number (from low to high)
    x_num_spheres = resolution

    # x offset per y row
    row_offset = sphere_radius

    y_num_spheres = math.floor(2.0 / (sphere_radius * math.sqrt(3)))

    # z offset per layer
    layer_z_offset = (2.0 * sphere_radius * math.sqrt(6)) / 3.0
    z_num_spheres = math.floor(2.0 / layer_z_offset)

    # x offset per layer
    layer_x_offset = sphere_radius

    # y offset per layer
    layer_y_offset = sphere_radius * math.sqrt(3) / 3.0

    sphere_positions = np.zeros((x_num_spheres, y_num_spheres, z_num_spheres), dtype=int)
    for z in range(z_num_spheres):
        for y in range(y_num_spheres):
            for x in range(x_num_spheres):
                # x,y,z is the center of the sphere
                # we use hexagonal packing, so the offset is different for even and odd rows and layers
                x_pos = (x * 2.0 * sphere_radius) + ((y % 2) * row_offset) + ((z % 2) * layer_x_offset)
                y_pos = y * sphere_radius * math.sqrt(3) + ((z % 2) * layer_y_offset)
                z_pos = z * layer_z_offset

                # put x,y,z into the correct range
                x_pos -= 1.0
                y_pos -= 1.0
                z_pos -= 1.0

                # check if the sphere is inside the mesh
                if f.contains([x_pos, y_pos, z_pos]):
                    sphere_positions[x, y, z] = 1
    
    sphere_defined = np.argwhere(sphere_positions)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    if (args.output_pc or args.output_mesh):
        # use the same x_pos, y_pos, z_pos as above to get the sphere center
        sphere_centers = np.zeros((sphere_defined.shape[0], 3))
        sphere_centers[:, 0] = sphere_defined[:, 0] * 2.0 * sphere_radius + (sphere_defined[:, 1] % 2) * row_offset + (sphere_defined[:, 2] % 2) * layer_x_offset
        sphere_centers[:, 1] = sphere_defined[:, 1] * sphere_radius * math.sqrt(3) + (sphere_defined[:, 2] % 2) * layer_y_offset
        sphere_centers[:, 2] = sphere_defined[:, 2] * layer_z_offset
        sphere_centers -= 1.0
        sphere_centers[:, 2] += 1.0

        if args.output_pc:
            sphere_centers = trimesh.points.PointCloud(sphere_centers)
            sphere_centers.export('sphere_centers.ply')

        # the output mesh file will be really big!!!
        if args.output_mesh:
            sphere = trimesh.creation.icosphere(radius=sphere_radius, subdivisions=2)
            sphere.apply_translation([0, 0, -1.0])
            sphere_vertices = np.zeros((sphere_centers.shape[0] * sphere.vertices.shape[0], 3))
            sphere_faces = np.zeros((sphere_centers.shape[0] * sphere.faces.shape[0], 3), dtype=int)
            for i in range(sphere_centers.shape[0]):
                sphere_vertices[i * sphere.vertices.shape[0]: (i + 1) * sphere.vertices.shape[0]] = sphere.vertices + sphere_centers[i]
                sphere_faces[i * sphere.faces.shape[0]: (i + 1) * sphere.faces.shape[0]] = sphere.faces + i * sphere.vertices.shape[0]
            
            sphere_mesh = trimesh.Trimesh(vertices=sphere_vertices, faces=sphere_faces)
            sphere_mesh.export('sphere_mesh.ply')
    
    if args.shadertoy_compatible:
        # iterate over sphere_positions grid and output the sphere centers in shadertoy-compatible format
        # using a vec3 for each sphere center is not necessary and just takes up space in shadertoy
        # instead, use the inherent hexagonal grid structure to compute the sphere center in the shader
        # for efficient space, we represent the grid with negative number for each sequence of 0s and positive number for each sequence of 1s
        # the shader will then iterate over the grid and compute the sphere center for each.

        total_spheres = np.sum(sphere_positions).astype(int)

        sphere_center_grid = []
        for z in range(z_num_spheres):
            sphere_center_grid.append([])
            for y in range(y_num_spheres):
                sphere_center_grid[z].append([])
                # number of spheres (or number of not spheres) until now in this row
                non_sphere_seq = 0
                sphere_seq = 0
                for x in range(x_num_spheres):
                    if sphere_positions[x, y, z] == 1:
                        if non_sphere_seq > 0:
                            sphere_center_grid[z][y].append(-non_sphere_seq)
                            non_sphere_seq = 0
                        sphere_seq += 1
                    else:
                        if sphere_seq > 0:
                            sphere_center_grid[z][y].append(sphere_seq)
                            sphere_seq = 0
                        non_sphere_seq += 1
                
                # the last sequence
                if sphere_seq > 0:
                    sphere_center_grid[z][y].append(sphere_seq)
                elif non_sphere_seq > 0:
                    sphere_center_grid[z][y].append(-non_sphere_seq)

        # print sphere_center_grid in shadertoy-compatible format
        print_string = 'const int spheres[] = int[] ( \n' 
        current_line_length = 0
        for z in range(z_num_spheres):
            for y in range(y_num_spheres):
                for x in range(len(sphere_center_grid[z][y])):
                    addition_to_print = str(sphere_center_grid[z][y][x]) + ','
                    current_line_length += len(addition_to_print)

                    # insert newline if the line is too long
                    if current_line_length > 80:
                        addition_to_print += '\n'
                        current_line_length = 0
                    print_string += addition_to_print
            if z == z_num_spheres - 1:
                if print_string[-1] == ',':
                    print_string = print_string[:-1]
                elif print_string[-2] == ',':
                    print_string = print_string[:-2]
        
        print_string = print_string + ');'

        print('Shadertoy Code: \n', print_string, sep='')
        print('Total number of spheres:', total_spheres)
        print('Sphere radius:', sphere_radius)
        print('x, y, z number of spheres in each dimension:', x_num_spheres, y_num_spheres, z_num_spheres)