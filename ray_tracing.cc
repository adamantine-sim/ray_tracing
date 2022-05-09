/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <ArborX.hpp>
#include <ArborX_Ray.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <cfloat>
#include <fstream>

template <typename MemorySpace>
struct TriangleBoundingVolume
{
  Kokkos::View<ArborX::Experimental::Triangle *, MemorySpace> triangles;
};

template <typename MemorySpace>
struct ArborX::AccessTraits<TriangleBoundingVolume<MemorySpace>,
                            ArborX::PrimitivesTag>
{
  using memory_space = MemorySpace;

  KOKKOS_FUNCTION static std::size_t
  size(const TriangleBoundingVolume<MemorySpace> &tbv)
  {
    return tbv.triangles.extent(0);
  }

  KOKKOS_FUNCTION static ArborX::Box
  get(TriangleBoundingVolume<MemorySpace> const &tbv, std::size_t const i)
  {
    auto const &triangle = tbv.triangles(i);
    float max_coord[3] = {triangle.a[0], triangle.a[1], triangle.a[2]};
    float min_coord[3] = {triangle.a[0], triangle.a[1], triangle.a[2]};
    for (int i = 0; i < 3; ++i)
    {
      // Max
      if (triangle.b[i] > max_coord[i])
        max_coord[i] = triangle.b[i];
      if (triangle.c[i] > max_coord[i])
        max_coord[i] = triangle.c[i];

      // Min
      if (triangle.b[i] < min_coord[i])
        min_coord[i] = triangle.b[i];
      if (triangle.c[i] < min_coord[i])
        min_coord[i] = triangle.c[i];
    }

    return {{min_coord[0], min_coord[1], min_coord[2]},
            {max_coord[0], max_coord[1], max_coord[2]}};
  }
};

template <typename MemorySpace>
struct Rays
{
  Kokkos::View<ArborX::Experimental::Ray *, MemorySpace> _rays;
};

template <typename MemorySpace>
struct ArborX::AccessTraits<Rays<MemorySpace>, ArborX::PredicatesTag>
{
  using memory_space = MemorySpace;

  KOKKOS_FUNCTION static std::size_t size(const Rays<MemorySpace> &rays)
  {
    return rays._rays.extent(0);
  }
  KOKKOS_FUNCTION static auto get(Rays<MemorySpace> const &rays, std::size_t i)
  {
    return attach(intersects(rays._rays(i)), (int)i);
  }
};

template <typename MemorySpace>
struct ProjectPointDistance
{
  Kokkos::View<ArborX::Experimental::Triangle *, MemorySpace> triangles;
  Kokkos::View<float *, MemorySpace> distance;

  template <typename Predicate>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate,
                                  int const primitive_index) const
  {
    auto const &ray = ArborX::getGeometry(predicate);
    float tmin;
    float tmax;
    int const i = getData(predicate);
    // intersects only if triangle is in front of the ray
    if (intersection(ray, triangles(primitive_index), tmin, tmax) && (tmax >= 0.f))
    {
      distance(i) = tmin;
    }
  }
};

ArborX::Point read_point(char *facet)
{
  char f1[4] = {facet[0], facet[1], facet[2], facet[3]};
  char f2[4] = {facet[4], facet[5], facet[6], facet[7]};
  char f3[4] = {facet[8], facet[9], facet[10], facet[11]};

  ArborX::Point vertex;
  vertex[0] = *(reinterpret_cast<float *>(f1));
  vertex[1] = *(reinterpret_cast<float *>(f2));
  vertex[2] = *(reinterpret_cast<float *>(f3));

  return vertex;
}

// Algorithm to read STL is based on http://www.sgh1.net/posts/read-stl-file.md
template <typename MemorySpace>
Kokkos::View<ArborX::Experimental::Triangle *, MemorySpace>
read_stl(std::string const &filename)
{
  std::ifstream file(filename.c_str(), std::ios::binary);
  ARBORX_ASSERT(file.good());

  // read 80 byte header
  char header_info[80] = "";
  file.read(header_info, 80);

  // Read the number of Triangle
  unsigned long long int n_triangles = 0;
  {
    char n_tri[4];
    file.read(n_tri, 4);
    n_triangles = *(reinterpret_cast<unsigned long long *>(n_tri));
  }

  Kokkos::View<ArborX::Experimental::Triangle *, MemorySpace> triangles(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "triangles"),
      n_triangles);

  auto triangles_host = Kokkos::create_mirror_view(triangles);

  for (unsigned int i = 0; i < n_triangles; ++i)
  {
    char facet[50];

    // read one 50-byte triangle
    file.read(facet, 50);

    // populate each point of the triangle
    // facet + 12 skips the triangle's unit normal
    auto p1 = read_point(facet + 12);
    auto p2 = read_point(facet + 24);
    auto p3 = read_point(facet + 36);

    // add a new triangle to the View
    triangles_host(i) = {p1, p2, p3};
  }
  file.close();

  Kokkos::deep_copy(triangles, triangles_host);

  return triangles;
}

template <typename MemorySpace>
Kokkos::View<ArborX::Experimental::Ray *, MemorySpace>
read_ray_file(std::string const &filename)
{
  std::ifstream file(filename.c_str());
  ARBORX_ASSERT(file.good());

  int n_rays = 0;
  file >> n_rays;

  Kokkos::View<ArborX::Experimental::Ray *, MemorySpace> rays(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "rays"), n_rays);

  auto rays_host = Kokkos::create_mirror_view(rays);

  for (int i = 0; i < n_rays; ++i)
  {
    float x, y, z, dir_x, dir_y, dir_z;
    file >> x >> y >> z >> dir_x >> dir_y >> dir_z;
    rays_host(i) = {{x, y, z}, {dir_x, dir_y, dir_z}};
  }
  file.close();

  Kokkos::deep_copy(rays, rays_host);

  return rays;
}

void outputTXT(Kokkos::View<ArborX::Point *, Kokkos::HostSpace> point_cloud,
               std::ostream &file, std::string const &delimiter)
{
  int const n_points = point_cloud.extent(0);
  for (int i = 0; i < n_points; ++i)
  {
    file << point_cloud(i)[0] << delimiter << point_cloud(i)[1] << delimiter
         << point_cloud(i)[2] << "\n";
  }
}

void outputVTK(
    Kokkos::View<ArborX::Point *, Kokkos::HostSpace> full_point_cloud,
    std::ostream &file)
{
  // For the vtk output, we remove all the points that are at infinity
  std::vector<ArborX::Point> point_cloud;
  for (unsigned int i = 0; i < full_point_cloud.extent(0); ++i)
  {
    if (full_point_cloud(i)[0] < FLT_MAX)
    {
      point_cloud.emplace_back(full_point_cloud(i)[0], full_point_cloud(i)[1],
                               full_point_cloud(i)[2]);
    }
  }

  // Write the header
  file << "# vtk DataFile Version 2.0\n";
  file << "Ray tracing\n";
  file << "ASCII\n";
  file << "DATASET POLYDATA\n";

  int const n_points = point_cloud.size();
  file << "POINTS " << n_points << " float\n";
  for (int i = 0; i < n_points; ++i)
  {
    file << point_cloud[i][0] << " " << point_cloud[i][1] << " "
         << point_cloud[i][2] << "\n";
  }

  // We need to associate a value to each point. We arbitrarly choose 1.
  file << "POINT_DATA " << n_points << "\n";
  file << "SCALARS value float 1\n";
  file << "LOOKUP_TABLE table\n";
  for (int i = 0; i < n_points; ++i)
  {
    file << "1.0\n";
  }
}

int main(int argc, char *argv[])
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;

  Kokkos::ScopeGuard guard(argc, argv);

  namespace bpo = boost::program_options;

  std::string stl_filename;
  std::string ray_filename;
  std::string output_filename;
  std::string output_type;
  int n_ray_files;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help,h", "help message" )
    ("stl_file,s", bpo::value<std::string>(&stl_filename), "name of the STL file")
    ("ray_files,r", bpo::value<std::string>(&ray_filename), "name of the ray file")
    ("n_ray_files,n", bpo::value<int>(&n_ray_files), "number of ray files")
    ("output_files,o", bpo::value<std::string>(&output_filename), "name of the output file")
    ("output_type,t", bpo::value<std::string>(&output_type), 
       "type of the output: csv, numpy, or vtk")
    ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help") > 0)
  {
    std::cout << desc << '\n';
    return 1;
  }

  // Read the STL file
  Kokkos::Profiling::pushRegion("ArborX::read_stl");
  auto triangles = read_stl<MemorySpace>(stl_filename);
  Kokkos::Profiling::popRegion();

  // Build the BVH
  ExecutionSpace exec_space;
  ArborX::BVH<MemorySpace> bvh(exec_space,
                               TriangleBoundingVolume<MemorySpace>{triangles});

  for (int i = 0; i < n_ray_files; ++i)
  {
    // Read the ray file
    std::string current_ray_filename =
        ray_filename + "-" + std::to_string(i) + ".txt";

    Kokkos::Profiling::pushRegion("ArborX::read_ray_files");
    auto rays = read_ray_file<MemorySpace>(current_ray_filename);
    Kokkos::Profiling::popRegion();
    unsigned int n_rays = rays.extent(0);

    Kokkos::View<float *, MemorySpace> distance(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "distance"), n_rays);
    Kokkos::deep_copy(distance, -1.);
    bvh.query(exec_space, Rays<MemorySpace>{rays},
              ProjectPointDistance<MemorySpace>{triangles, distance});

    Kokkos::View<ArborX::Point *, MemorySpace> point_cloud("point_cloud",
                                                           n_rays);
    Kokkos::parallel_for(
        "project_points", Kokkos::RangePolicy<ExecutionSpace>(0, n_rays),
        KOKKOS_LAMBDA(int i) {
          if (distance(i) >= 0)
          {
            point_cloud(i) = {
                rays(i)._origin[0] + distance(i) * rays(i)._direction[0],
                rays(i)._origin[1] + distance(i) * rays(i)._direction[1],
                rays(i)._origin[2] + distance(i) * rays(i)._direction[2]};
          }
          else
          {
            float max_value = FLT_MAX;
            point_cloud(i) = {max_value, max_value, max_value};
          }
        });

    auto point_cloud_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, point_cloud);

    // Write results to file
    Kokkos::Profiling::pushRegion("ArborX::write_results");
    if ((output_type == "csv") || (output_type == "numpy"))
    {
      std::string delimiter = " ";
      if (output_type == "csv")
        delimiter = ",";
      else
        delimiter = " ";
      std::ofstream file;
      file.open(output_filename + "-" + std::to_string(i) + ".txt");
      outputTXT(point_cloud_host, file, delimiter);
      file.close();
    }
    else
    {
      std::ofstream file;
      file.open(output_filename + "-" + std::to_string(i) + ".vtk");
      outputVTK(point_cloud_host, file);
      file.close();
    }
    Kokkos::Profiling::popRegion();
  }
}

