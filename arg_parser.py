import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Hexagonal Sphere Packing')

    parser.add_argument('--input', type=str, default='./', help='Input mesh path')
    parser.add_argument('--output', type=str, default='./', help='Output folder')
    parser.add_argument('--resolution', type=int, default=25, help='Number of spheres along the largest dimension')
    parser.add_argument('--sdf-resolution', type=int, default=256, help='Resolution of the SDF grid, should be larger than resolution')
    parser.add_argument('--shadertoy-compatible', action='store_true', help='Output in shadertoy-compatible format')
    parser.add_argument('--output-pc', action='store_true', help='Output a point cloud for visualization')
    parser.add_argument('--output-mesh', action='store_true', help='Output a mesh for visualization')

    return parser.parse_args()