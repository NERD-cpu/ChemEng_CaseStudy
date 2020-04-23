"""

CASE STUDY 1: Selective oxidation of o-xylene to phthalic anhydride.

Highlights:
-Convection-diffusion-reaction simulation in single packed-bed tube.
-Two-dimensional pseudo homogeneous mathematical model.
-Five PDEs accounting for species mass equations and reactor temperature.
-One ODE accounting for the cooling jacket model.
-Physical properties assumed to be constant.
-Catalytic rate model for V2O5 assumed of pseudo-first-order.

A + 3.B --> C + 3.D

A: o-Xylene
B: Oxygen
C: Phthalic anhydride
D: Water

"""

# FEniCS Project Packages
from dolfin import XDMFFile, Mesh, FiniteElement, triangle, MixedElement, \
                   FunctionSpace, Measure, Constant, TestFunctions, \
                   Function, split, Expression, project, dot, grad, dx, \
                   solve, as_vector, MeshValueCollection, cpp
from ufl.operators import exp

# Mesh format conversion
import meshio

# General-purpose packages in computer science.
import os
import numpy as np
import math as mt

# Time-stepping
t = 0.0
tf = 5.0                     # Final time, sec
num_steps = 100              # Number of time steps
delta_t = tf / num_steps     # Time step size, sec
dt = Constant(delta_t)

R_path = 'Results'
Writting_xdmf = True  # Data storage for later visualization.


"""Gmsh mesh format conversion by package Meshio."""
Wri_path = "Gmsh_meshes"
msh = meshio.read(os.path.join(Wri_path, 'SinglePBR.msh'))

line_cells = []
for cell in msh.cells:
    if cell.type == "triangle":
        triangle_cells = cell.data
    elif  cell.type == "line":
        if len(line_cells) == 0:
            line_cells = cell.data
        else:
            line_cells = np.vstack([line_cells, cell.data])

line_data = []
for key in msh.cell_data_dict["gmsh:physical"].keys():
    if key == "line":
        if len(line_data) == 0:
            line_data = msh.cell_data_dict["gmsh:physical"][key]
        else:
            line_data = np.vstack([line_data, msh.cell_data_dict["gmsh:physical"][key]])
    elif key == "triangle":
        triangle_data = msh.cell_data_dict["gmsh:physical"][key]


triangle_mesh = meshio.Mesh(points=msh.points,
                            cells={"triangle": triangle_cells})
line_mesh = meshio.Mesh(points=msh.points,
                        cells=[("line", line_cells)],
                        cell_data={"name_to_read":[line_data]})

meshio.write(os.path.join(Wri_path, "md_.xdmf"), triangle_mesh)
meshio.xdmf.write(os.path.join(Wri_path, "mf_.xdmf"), line_mesh)

# Reading mesh data stored in .xdmf files.
mesh = Mesh()
with XDMFFile(os.path.join(Wri_path, "md_.xdmf")) as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 1)

with XDMFFile(os.path.join(Wri_path, "mf_.xdmf")) as infile:
    infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)


# Define function spaces for PDEs variational formulation.
P1 = FiniteElement('P', mesh.ufl_cell(), 1)  # Lagrange 1-order polynomials family
element = MixedElement([P1, P1, P1, P1])
V = FunctionSpace(mesh, element)  # Test functions
function_space = FunctionSpace(mesh, P1)

# Splitting test and trial functions
v_A, v_B, v_C, v_T = TestFunctions(V)
u = Function(V)
u_n = Function(V)
u_A, u_B, u_C, u_T = split(u)


# Retrieve boundaries marks for Robin boundary conditions.
ds_in = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=1)
ds_wall = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=2)
