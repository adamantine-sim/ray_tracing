Ray Tracing
===========
This codes performs ray tracing and computes the points where the rays intersect
a geometry described by a stl file.

Installation
------------
The depends on [boost](https://www.boost.org), [Kokkos](https://github.com/kokkos/kokkos), 
and [ArborX](https://github.com/arborx/AborX). You can configure `ray_tracing`
using:
```
cmake \
  -D CMAKE_CXX_COMPILER=${KOKKOS_INSTALL_DIR}/bin/nvcc_wrapper \ 
  -D CMAKE_CXX_EXTENSIONS=OFF \
  -D CMAKE_PREFIX_PATH="${KOKKOS_INSTALL_DIR};${ARBORX_INSTALL_DIR};${BOOST_INSTALL_DIR}" \
${RAY_TRACING_SOURCE_DIR}
```
`nvcc_wrapper` is only necessary when using `nvcc`.

Usage
-----
The code expects the following options:
 * -s: the name of the STL file that contains the geometry
 * -r: prefix of the ray file. The file name is expected to be of the form:
 ray\_prefix-i.txt with `i` a number starting at zero. 
 * -n: number of ray files.
 * -o: prefix of the output files. The file name is of the form
 output\_prefix-i.txt or output\_prefix-i.vtk
 * -t: type of the output: csv, numpy (the array can be loaded using loadtxt),
 or vtk.

The format of the ray file is the following:
```
Number of rays
Coordinates of the ray origin  Direction of the ray
```
In practice, a file with three rays starting in (0,0,0) and following the axes
`x`, `y`, and `z` will look as follows
```
3
0. 0. 0. 1. 0. 0.
0. 0. 0. 0. 1. 0.
0. 0. 0. 0. 0. 1.
```
