#include <iostream>
#include <utility>
#include <array>
#include <fstream>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <bitset>

#define CGAL_EIGEN3_ENABLED
//#define DETERMINE_VOXEL_SIZE
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/boost/graph/convert_nef_polyhedron_to_polygon_mesh.h>
#include <CGAL/Optimal_bounding_box/oriented_bounding_box.h>
#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/polygon_mesh_to_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/repair_degeneracies.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_length_cost.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_length_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Midpoint_placement.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Polyhedral_envelope_filter.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Bounded_normal_change_filter.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/GarlandHeckbert_policies.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Bounded_normal_change_placement.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Constrained_placement.h>

namespace PMP = CGAL::Polygon_mesh_processing;
namespace SMS = CGAL::Surface_mesh_simplification;

typedef CGAL::Exact_predicates_exact_constructions_kernel			Exact_Kernel;
typedef CGAL::Exact_predicates_inexact_constructions_kernel			Inexact_Kernel;
typedef CGAL::Polyhedron_3<Inexact_Kernel>							Inexact_Polyhedron;
typedef CGAL::Polyhedron_3<Exact_Kernel>							Exact_Polyhedron;
typedef CGAL::Nef_polyhedron_3<Exact_Kernel>						Nef_polyhedron;
typedef Exact_Kernel::Point_3										ExactPoint;
typedef Inexact_Kernel::Point_3										InexactPoint;
typedef Inexact_Kernel::Triangle_3									Triangle;
typedef Exact_Kernel::Vector_3										ExactVector;
typedef CGAL::Surface_mesh<InexactPoint>							Mesh;
typedef CGAL::Surface_mesh<ExactPoint>								ExactMesh;
typedef CGAL::AABB_face_graph_triangle_primitive<Exact_Polyhedron>	Primitive;
typedef CGAL::AABB_traits<Exact_Kernel, Primitive>					Traits;
typedef CGAL::AABB_tree<Traits>										Tree;
typedef CGAL::Side_of_triangle_mesh<Exact_Polyhedron, Exact_Kernel>	Point_inside;
typedef SMS::Polyhedral_envelope_filter<Inexact_Kernel,
	SMS::Bounded_normal_change_filter<>>							Filter;

typedef boost::graph_traits<Inexact_Polyhedron>::edge_descriptor      edge_descriptor;
typedef boost::graph_traits<Inexact_Polyhedron>                             GraphTraits;
typedef typename GraphTraits::halfedge_descriptor                           halfedge_descriptor;
typedef typename GraphTraits::vertex_descriptor                             vertex_descriptor;
typedef Exact_Polyhedron::Vertex_handle Vertex;
typedef Exact_Polyhedron::Face_handle Face;
typedef std::vector<bool> VOXEL_GRID;
typedef std::vector<VOXEL_GRID>	MIPMAP_TYPE;

#define BASE_RESOLUTION 256

typedef struct {
	int level;
	unsigned int pos;
} Node;

#ifdef DETERMINE_VOXEL_SIZE
double determine_size(double s_o_sum, double s_f, double s_c, double mu_0, double mu_1)
{
	auto s_v = s_f;
	auto f = s_o_sum / s_v;
	std::cout << "f(s_v)=" << f << "\n";
	if (mu_0 <= f && f <= mu_1)
	{
		return s_v;
	}
	else if (f < mu_0)
	{
		s_f = s_v;
	}
	else if (f > mu_1)
	{
		s_c = s_v;
	}
	return (s_c + s_f) / 2.;
}
#endif

unsigned int coords_to_voxel_idx(unsigned int x_idx, unsigned int y_idx, unsigned int z_idx, std::array<unsigned int, 3> numVoxels)
{
	return z_idx + y_idx * numVoxels[2] + x_idx * numVoxels[2] * numVoxels[1];
}

unsigned int coords_to_voxel_idx(unsigned int x_idx, unsigned int y_idx, unsigned int z_idx, int numVoxels)
{
	return z_idx + y_idx * numVoxels + x_idx * numVoxels * numVoxels;
}

void convert_voxel_idx_to_coords(unsigned int idx, unsigned int numVoxels, unsigned int& x_idx, unsigned int& y_idx, unsigned int& z_idx)
{
	x_idx = idx / (numVoxels * numVoxels);
	auto const w = idx % (numVoxels * numVoxels);
	y_idx = w / numVoxels;
	z_idx = w % numVoxels;
}

std::array<ExactPoint, 8> calc_voxel_points(unsigned int idx, std::array<unsigned int, 3> numVoxels, ExactPoint min_point,
	const std::array<ExactVector, 3>& voxel_strides, bool* new_scanline = nullptr)
{
	unsigned int x_idx, y_idx, z_idx;

	convert_voxel_idx_to_coords(idx, numVoxels[0], x_idx, y_idx, z_idx);

	if (new_scanline)
	{
		*new_scanline = z_idx == 0;
	}

	return {
		min_point + x_idx * voxel_strides[0] + y_idx * voxel_strides[1] + z_idx * voxel_strides[2],
		min_point + (x_idx + 1u) * voxel_strides[0] + y_idx * voxel_strides[1] + z_idx * voxel_strides[2],
		min_point + x_idx * voxel_strides[0] + (y_idx + 1u) * voxel_strides[1] + z_idx * voxel_strides[2],
		min_point + (x_idx + 1u) * voxel_strides[0] + (y_idx + 1u) * voxel_strides[1] + z_idx * voxel_strides[2],
		min_point + x_idx * voxel_strides[0] + y_idx * voxel_strides[1] + (z_idx + 1u) * voxel_strides[2],
		min_point + (x_idx + 1u) * voxel_strides[0] + y_idx * voxel_strides[1] + (z_idx + 1u) * voxel_strides[2],
		min_point + x_idx * voxel_strides[0] + (y_idx + 1u) * voxel_strides[1] + (z_idx + 1u) * voxel_strides[2],
		min_point + (x_idx + 1u) * voxel_strides[0] + (y_idx + 1u) * voxel_strides[1] + (z_idx + 1u) * voxel_strides[2]
	};
}

// make a voxel with 4 tetrahedron?
void calc_voxel_from_idx_tets(unsigned int idx, std::array<unsigned int, 3> numVoxels, ExactPoint min_point,
	const std::array<ExactVector, 3>& voxel_strides, Exact_Polyhedron& voxel, bool* new_scanline)
{
	auto const p = calc_voxel_points(idx, numVoxels, min_point, voxel_strides, new_scanline);

	voxel.make_tetrahedron(p[0], p[3], p[5], p[1]);
	voxel.make_tetrahedron(p[0], p[3], p[2], p[6]);
	voxel.make_tetrahedron(p[0], p[4], p[5], p[6]);
	voxel.make_tetrahedron(p[5], p[6], p[7], p[3]);
	assert(voxel.is_valid());
}

void calc_voxel_from_idx_hex(unsigned int idx, std::array<unsigned int, 3> numVoxels, ExactPoint min_point,
	const std::array<ExactVector, 3>& voxel_strides, Exact_Polyhedron& voxel)
{
	auto const p = calc_voxel_points(idx, numVoxels, min_point, voxel_strides);

	unsigned int x_idx, y_idx, z_idx;
	convert_voxel_idx_to_coords(idx, numVoxels[0], x_idx, y_idx, z_idx);

	//std::cout << idx << "th voxel(" << x_idx << ", " << y_idx << ", " << z_idx << ")\n";
	CGAL::make_hexahedron(p[0], p[1], p[3], p[2], p[6], p[4], p[5], p[7], voxel);
	assert(voxel.is_valid());
}

float base_cellsize;
ExactPoint global_min_point;
int data_res[3];
std::vector<bool> voxelize(Mesh surface, Exact_Polyhedron poly)
{

	float se_size = 0.1f;
	float margin = se_size + 2.0f / (2 * BASE_RESOLUTION);

	Tree mesh_tree(faces(poly).first, faces(poly).second, poly);
	//Tree mesh_tree(faces(surface).first, faces(surface).second, surface);
	CGAL::Bbox_3 bbox_origin = mesh_tree.bbox();
	float longest_axis = std::max(bbox_origin.xmax() - bbox_origin.xmin(),
		std::max(bbox_origin.ymax() - bbox_origin.ymin(), bbox_origin.zmax() - bbox_origin.zmin()));
	float thickness = longest_axis / BASE_RESOLUTION;
	float offset = 2 * thickness + margin;
	float axis_len = longest_axis + 2 * offset;
	float cell_size = axis_len / BASE_RESOLUTION;

	float new_xmin = bbox_origin.xmin() - offset;
	float new_ymin = bbox_origin.ymin() - offset;
	float new_zmin = bbox_origin.zmin() - offset;

	CGAL::Bbox_3 grid_aabb(
		new_xmin,
		new_ymin,
		new_zmin,
		new_xmin + axis_len,
		new_ymin + axis_len,
		new_zmin + axis_len
	);

	ExactPoint grid_min(grid_aabb.xmin(), grid_aabb.ymin(), grid_aabb.zmin());

	// index where the actual data is finished at each axis
	int data_resolution[3] = {
		static_cast<int>(ceil((bbox_origin.xmax() - grid_aabb.xmin()) / cell_size)),
		static_cast<int>(ceil((bbox_origin.ymax() - grid_aabb.ymin()) / cell_size)),
		static_cast<int>(ceil((bbox_origin.zmax() - grid_aabb.zmin()) / cell_size))
	};
	data_res[0] =
		static_cast<int>(ceil((bbox_origin.xmax() - grid_aabb.xmin()) / cell_size)),
		data_res[1] =
		static_cast<int>(ceil((bbox_origin.ymax() - grid_aabb.ymin()) / cell_size)),
		data_res[2] =
		static_cast<int>(ceil((bbox_origin.zmax() - grid_aabb.zmin()) / cell_size));
	std::cout << "data resolution " << data_resolution[0] << ", " << data_resolution[1] << ", " << data_resolution[2] << "\n";

	std::array<unsigned int, 3> numVoxels = { BASE_RESOLUTION, BASE_RESOLUTION, BASE_RESOLUTION };
	std::array<ExactVector, 3> voxel_strides = { ExactVector(cell_size, 0, 0),
		ExactVector(0, cell_size, 0), ExactVector(0, 0, cell_size) };

	std::vector<unsigned int> intersecting_voxels;

	auto numVoxel = pow(BASE_RESOLUTION, 3);// numVoxels[0] * numVoxels[1] * numVoxels[2];
	bool interior = false;
	unsigned int last_voxel = 0;
	std::vector<bool> voxels_marking(numVoxel, false); // either outside 0, surface 1 or interior 2

	Tree tree(faces(poly).first, faces(poly).second, poly);
	Point_inside inside_tester(tree);

	// Exact_Polyhedron voxels;
	std::cout << "Check for intersection\n";
	for (unsigned int i = 0; i < data_resolution[0]; i++) {
		for (unsigned int j = 0; j < data_resolution[1]; j++) {
			for (unsigned int k = 0; k < data_resolution[2]; k++) {

				unsigned int idx = i * pow(BASE_RESOLUTION, 2) + j * BASE_RESOLUTION + k;

				Exact_Polyhedron voxel = Exact_Polyhedron();
				bool new_scanline;

				calc_voxel_from_idx_tets(idx, numVoxels, grid_min, voxel_strides, voxel, &new_scanline);

				if (CGAL::Polygon_mesh_processing::do_intersect(voxel, poly))
				{
					intersecting_voxels.push_back(idx);
					voxels_marking[idx] = true;
					continue;
				}

				bool inside = true;

				for (auto vert : voxel.vertex_handles())
				{
					if (inside_tester(vert->point()) != CGAL::ON_BOUNDED_SIDE)
					{
						inside = false;
					}
				}

				if (inside)
				{
					intersecting_voxels.push_back(idx);
					voxels_marking[idx] = true;
				}
				//if(k % 5 == 0) std::cout << "k : " << k << "\n";
			}
			//if(j % 5 == 0)  std::cout << "j : " << j << "\n";
		}
		//if (i % 5 == 0) std::cout << "i : " << i << "\n";
	}

	// for mesh generation of voxel
	/*Exact_Polyhedron voxels;
	for (auto voxel_idx : intersecting_voxels)
	{
		calc_voxel_from_idx_hex(voxel_idx, numVoxels, grid_min, voxel_strides, voxels);
	}*/
	global_min_point = grid_min;
	base_cellsize = cell_size;
	//CGAL::write_off("C:/Users/jinjo/Documents/TUD/a2024wise/VC_lab/CageModeler/models/cactus_voxel.off", voxels);

	return voxels_marking;
}
void draw_voxel(int depth, VOXEL_GRID grid, int resol, std::string obj) {
	Exact_Polyhedron voxels;
	std::array<unsigned int, 3> numVoxels = { resol, resol, resol };
	std::array<ExactVector, 3> voxel_strides = {
		ExactVector(base_cellsize * pow(2, std::max(depth, 0)), 0, 0),
		ExactVector(0, base_cellsize * pow(2, std::max(depth, 0)), 0),
		ExactVector(0, 0, base_cellsize * pow(2, std::max(depth,0)))
	};

	for (int i = 0; i < grid.size(); i++)
	{
		//unsigned int x, y, z;
		//convert_voxel_idx_to_coords(i, resol, x, y, z);
		//if (y < (int)(BASE_RESOLUTION / 2)) continue;

		if (grid[i])
			calc_voxel_from_idx_hex(i, numVoxels, global_min_point, voxel_strides, voxels);
	}
	std::string filename = "C:/Users/jinjo/Documents/TUD/a2024wise/VC_lab/CageModeler/models/mipmap_data/";
	filename.append(obj);
	filename.append("_");
	if (depth == -1) filename.append("dilated");
	else if (depth == -2) filename.append("contour");
	else if (depth == -3) filename.append("eroded");
	else if (depth == 0) filename.append("voxel");
	else filename.append(std::to_string(depth));
	filename.append("_RESOL");
	filename.append(std::to_string(BASE_RESOLUTION));
	filename.append(".off");
	CGAL::write_off(filename.c_str(), voxels);

}
bool check_8cube(int x, int y, int z, VOXEL_GRID prev_grid, int prev_resol) {
	bool result = false;
	for (int x_offset = 0; x_offset < 2; x_offset++) {
		for (int y_offset = 0; y_offset < 2; y_offset++) {
			for (int z_offset = 0; z_offset < 2; z_offset++) {
				int flat_idx = (2 * x + x_offset) * prev_resol * prev_resol +
					(2 * y + y_offset) * prev_resol + 2 * z + z_offset;
				result = (result || prev_grid[flat_idx]);
			}
		}
	}
	if (result) {
		int a = 3;
	}
	return result;
}
std::vector<Node> find_subcells(Node parent, MIPMAP_TYPE mipmap) {

	std::vector<Node> subcells;
	unsigned int parent_x, parent_y, parent_z;
	int parent_resol = BASE_RESOLUTION / pow(2, parent.level);

	convert_voxel_idx_to_coords(parent.pos, parent_resol, parent_x, parent_y, parent_z);

	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			for (int k = 0; k < 2; k++) {
				unsigned int idx =
					coords_to_voxel_idx(
						2 * parent_x + i,
						2 * parent_y + j,
						2 * parent_z + k,
						parent_resol * 2
					);

				subcells.push_back({ parent.level - 1, idx });
			}
	return subcells;
}
MIPMAP_TYPE generate_mipmap(VOXEL_GRID grid) {
	MIPMAP_TYPE mm_pyramid;
	mm_pyramid.push_back(grid);

	int total_depth = log2(BASE_RESOLUTION) + 1;

	for (int depth = 1; depth < total_depth; depth++) {
		int resol = BASE_RESOLUTION / pow(2, depth);
		VOXEL_GRID mipmap(pow(resol, 3), false);

		for (int i = 0; i < resol; i++) {
			for (int j = 0; j < resol; j++) {
				for (int k = 0; k < resol; k++) {
					int flat_idx = i * resol * resol + j * resol + k;
					mipmap[flat_idx] = check_8cube(i, j, k, mm_pyramid[depth - 1], resol * 2);
				}
			}
		}

		//draw_voxel(depth, mipmap, resol);
		std::cout << "just drew mipmap level " << depth << "\n";
		mm_pyramid.push_back(mipmap);
	}
	return mm_pyramid;
}

typedef struct {
	ExactPoint center;
	float radius;
	CGAL::Bbox_3 bbox;
	bool sphere;

}SE;

typedef struct {
	float scale;
	bool occupied;
} CONTOUR_ELEMENT;

CGAL::Bbox_3 calc_voxel_bbox(unsigned int idx, int resol, ExactPoint min_point,
	float cell_size)
{
	unsigned int x_idx, y_idx, z_idx;

	convert_voxel_idx_to_coords(idx, resol, x_idx, y_idx, z_idx);
	float x_min = CGAL::to_double(min_point.x()) + x_idx * cell_size;
	float y_min = CGAL::to_double(min_point.y()) + y_idx * cell_size;
	float z_min = CGAL::to_double(min_point.z()) + z_idx * cell_size;
	return CGAL::Bbox_3(
		x_min,
		y_min,
		z_min,
		x_min + cell_size,
		y_min + cell_size,
		z_min + cell_size
	);
}

void define_se(Node cell, float radius, SE& se, bool sphere) {

	CGAL::Bbox_3 voxel_bbox = calc_voxel_bbox(cell.pos, BASE_RESOLUTION / pow(2, cell.level), global_min_point, base_cellsize * pow(2, cell.level));
	se.center = ExactPoint(
		(voxel_bbox.xmax() + voxel_bbox.xmin()) / 2.0,
		(voxel_bbox.ymax() + voxel_bbox.ymin()) / 2.0,
		(voxel_bbox.zmax() + voxel_bbox.zmin()) / 2.0
	);
	se.radius = radius;
	float xmin = CGAL::to_double(se.center.x()) - radius;
	float ymin = CGAL::to_double(se.center.y()) - radius;
	float zmin = CGAL::to_double(se.center.z()) - radius;
	se.bbox = CGAL::Bbox_3(
		xmin,
		ymin,
		zmin,
		xmin + radius * 2,
		ymin + radius * 2,
		zmin + radius * 2
	);
	se.sphere = sphere;
	return;
}

float get_shortest_dist(ExactPoint point, CGAL::Bbox_3 cell) {
	double sphere_x = CGAL::to_double(point.x());
	double sphere_y = CGAL::to_double(point.y());
	double sphere_z = CGAL::to_double(point.z());
	float closest_x = std::max(cell.xmin(), std::min(sphere_x, cell.xmax()));
	float closest_y = std::max(cell.ymin(), std::min(sphere_y, cell.ymax()));
	float closest_z = std::max(cell.zmin(), std::min(sphere_z, cell.zmax()));

	// Calculate the distance from the sphere's center to the closest point on the bounding box
	float dist = pow(sphere_x - closest_x, 2) + pow(sphere_y - closest_y, 2) + pow(sphere_z - closest_z, 2);
	dist = std::sqrt(dist);

	return dist;

}

bool does_overlap(Node cell, SE& se) {
	int resol = BASE_RESOLUTION / pow(2, cell.level);
	float cell_size = base_cellsize * pow(2, cell.level);
	CGAL::Bbox_3 cell_bbox = calc_voxel_bbox(cell.pos, resol, global_min_point, cell_size);

	if (se.bbox.xmax() >= cell_bbox.xmin() &&
		se.bbox.ymax() >= cell_bbox.ymin() &&
		se.bbox.zmax() >= cell_bbox.zmin() &&
		se.bbox.xmin() <= cell_bbox.xmax() &&
		se.bbox.ymin() <= cell_bbox.ymax() &&
		se.bbox.zmin() <= cell_bbox.zmax()
		) {
		if (se.sphere) {
			float shortest_dist = get_shortest_dist(se.center, cell_bbox);
			if (shortest_dist > se.radius) {
				return false;
			}
			else {
				return true;
			}
		}
		else return true;
	}
	else return false;
}

VOXEL_GRID executeDilation(MIPMAP_TYPE mipmap) {

	VOXEL_GRID d_grid = mipmap[0];

	int mipmap_depth = mipmap.size();

	std::stack<Node> node_stack;
	std::array<unsigned int, 3> num_voxels = { BASE_RESOLUTION, BASE_RESOLUTION, BASE_RESOLUTION };
	for (int x = 0; x < BASE_RESOLUTION; x++) {
		if (x % 5 == 0) std::cout << "x : " << x << "\n";
		for (int y = 0; y < BASE_RESOLUTION; y++) {
			for (int z = 0; z < BASE_RESOLUTION; z++) {
				int flat_idx = coords_to_voxel_idx(x, y, z, num_voxels);
				if (d_grid[flat_idx] == true) continue;
				SE se;
				Node current_point = { 0, flat_idx };
				define_se(current_point, base_cellsize, se, false);
				//define_se(current_point, base_cellsize * 0.6, se, true);
				node_stack.push({ mipmap_depth - 1, 0 });
				while (!node_stack.empty() && d_grid[flat_idx] == false) {
					auto top_node = node_stack.top();
					node_stack.pop();

					if (top_node.level == 0) {
						d_grid[flat_idx] = true;
						//d_grid[top_node.pos] = true;
						std::stack<Node> empty_stack;
						node_stack.swap(empty_stack);
						break;
					}
					else {
						std::vector<Node> subcells = find_subcells(top_node, mipmap);
						for (auto& subcell : subcells) {
							bool subcell_val = mipmap[subcell.level][subcell.pos];
							bool overlap = does_overlap(subcell, se);
							if (subcell_val == true && does_overlap(subcell, se)) {
								node_stack.push(subcell);
							}
						}
					}
				}
			}
		}
	}
	return d_grid;
}

//utils for saving file
void saveExactPoint(const ExactPoint& point, std::ofstream& file) {
	// Serialize the x, y, and z coordinates of the point as doubles
	double x = CGAL::to_double(point.x());
	double y = CGAL::to_double(point.y());
	double z = CGAL::to_double(point.z());

	// Write the coordinates to the file
	file.write(reinterpret_cast<const char*>(&x), sizeof(x));
	file.write(reinterpret_cast<const char*>(&y), sizeof(y));
	file.write(reinterpret_cast<const char*>(&z), sizeof(z));
}

void loadExactPoint(ExactPoint& point, std::ifstream& file) {
	double x, y, z;

	// Read the coordinates from the file
	file.read(reinterpret_cast<char*>(&x), sizeof(x));
	file.read(reinterpret_cast<char*>(&y), sizeof(y));
	file.read(reinterpret_cast<char*>(&z), sizeof(z));

	// Create a new ExactPoint with the loaded coordinates
	point = ExactPoint(x, y, z);
}

void saveBoolVector(const std::vector<bool>& vec, std::ofstream& file) {
	size_t size = vec.size();
	file.write(reinterpret_cast<const char*>(&size), sizeof(size)); // Write size of vector

	unsigned char byte = 0;
	size_t bitCount = 0;

	for (size_t i = 0; i < size; ++i) {
		if (vec[i]) {
			byte |= (1 << bitCount);
		}

		++bitCount;
		if (bitCount == 8) {
			file.put(byte);
			byte = 0;
			bitCount = 0;
		}
	}

	if (bitCount > 0) {
		file.put(byte);
	}
}

std::vector<bool> loadBoolVector(std::ifstream& file) {
	size_t size;
	file.read(reinterpret_cast<char*>(&size), sizeof(size)); // Read the size of the vector

	std::vector<bool> vec(size);
	unsigned char byte = 0;
	size_t bitCount = 0;

	for (size_t i = 0; i < size; ++i) {
		if (bitCount == 0) {
			file.get(reinterpret_cast<char&>(byte));
		}

		vec[i] = (byte & (1 << bitCount)) != 0;
		++bitCount;
		if (bitCount == 8) {
			bitCount = 0;
		}
	}

	return vec;
}

void saveBoolVector2D(const std::vector<std::vector<bool>>& vec2D, const std::string& filename, float extraFloat, const ExactPoint& point) {
	std::ofstream file(filename, std::ios::binary);

	if (!file) {
		std::cerr << "Error opening file for writing!" << std::endl;
		return;
	}

	size_t outerSize = vec2D.size();
	file.write(reinterpret_cast<const char*>(&outerSize), sizeof(outerSize)); // Write size of outer vector

	for (const auto& innerVec : vec2D) {
		saveBoolVector(innerVec, file); // Save each inner vector
	}

	// Save the additional float value
	file.write(reinterpret_cast<const char*>(&extraFloat), sizeof(extraFloat));

	// Save the ExactPoint
	saveExactPoint(point, file);

	file.close();
}

void loadBoolVector2D(std::vector<std::vector<bool>>& vec2D, const std::string& filename, float& extraFloat, ExactPoint& point) {
	std::ifstream file(filename, std::ios::binary);

	if (!file) {
		std::cerr << "Error opening file for reading!" << std::endl;
		return;
	}

	size_t outerSize;
	file.read(reinterpret_cast<char*>(&outerSize), sizeof(outerSize)); // Read the size of outer vector

	vec2D.resize(outerSize);
	for (size_t i = 0; i < outerSize; ++i) {
		vec2D[i] = loadBoolVector(file); // Load each inner vector
	}

	// Load the additional float value
	file.read(reinterpret_cast<char*>(&extraFloat), sizeof(extraFloat));

	// Load the ExactPoint
	loadExactPoint(point, file);

	file.close();
}

bool areIdentical(const std::vector<std::vector<bool>>& vec1, const std::vector<std::vector<bool>>& vec2) {
	// First, check if the sizes of the outer vectors are the same
	if (vec1.size() != vec2.size()) {
		return false;
	}

	// Now, compare each inner vector
	for (size_t i = 0; i < vec1.size(); ++i) {
		// Check if the sizes of the inner vectors are the same
		if (vec1[i].size() != vec2[i].size()) {
			return false;
		}

		// Compare each element in the inner vectors
		for (size_t j = 0; j < vec1[i].size(); ++j) {
			if (vec1[i][j] != vec2[i][j]) {
				return false; // Found a mismatch
			}
		}
	}

	// If all checks passed, the vectors are identical
	return true;
}


int generate_save(std::string obj_name) {
	std::string filename_base = "C:/Users/jinjo/Documents/TUD/a2024wise/VC_lab/CageModeler/models/";
	//const std::string filename = "C:/Users/jinjo/Documents/TUD/a2024wise/VC_lab/CageModeler/models/cactus.obj";
	std::string filename_in = filename_base + obj_name + ".obj";
	std::string filename_out = filename_base + "mipmap_data/" + obj_name + "_mipmap_RESOL" + std::to_string(BASE_RESOLUTION) + ".bin";
	std::cout << "Loading surface\n";
	Exact_Polyhedron poly;
	if (!PMP::IO::read_polygon_mesh(filename_in, poly) || !CGAL::is_triangle_mesh(poly))
	{
		std::cerr << "Invalid input.\n";
		return EXIT_FAILURE;
	}
	Mesh surface;
	if (!CGAL::Polygon_mesh_processing::IO::read_polygon_mesh(filename_in, surface) || surface.is_empty())
	{
		std::cerr << "Invalid input file.\n";
		return EXIT_FAILURE;
	}
	int t1 = clock();
	auto voxel_grid = voxelize(surface, poly);
	std::cout << "voxelize done\n";
	int t2 = clock();

	draw_voxel(0, voxel_grid, BASE_RESOLUTION, obj_name);

	int t3 = clock();
	std::cout << "start generating mipmap\n";
	MIPMAP_TYPE mipmap_pyramid = generate_mipmap(voxel_grid);
	int t4 = clock();
	std::cout << "mipmap done. start dilation\n";
	printf("Voxelize elapsed time: %5.3f sec\n", float(t2 - t1) / CLOCKS_PER_SEC);
	printf("Drawing voxel elapsed time: %5.3f sec\n", float(t3 - t2) / CLOCKS_PER_SEC);
	printf("Generate Mipmap elapsed time: %5.3f sec\n", float(t4 - t3) / CLOCKS_PER_SEC);
	printf("Total elapsed time (from voxelization to mipmap): %5.3f sec\n", float(t4 - t1) / CLOCKS_PER_SEC);


	saveBoolVector2D(mipmap_pyramid, filename_out, base_cellsize, global_min_point);
}

bool check_neighbor(VOXEL_GRID& d_grid, char axis, int center_x, int center_y, int center_z) {
	int radius = 1;

	unsigned int center_idx = coords_to_voxel_idx(center_x, center_y, center_z, BASE_RESOLUTION);
	bool center_val = d_grid[center_idx];
	bool result = false;

	// check for z-1, z+1
	for (int offset = -radius; offset <= radius; offset++) {

		if (offset == 0) continue;

		int x = center_x, y = center_y, z = center_z;

		if (axis == 'x') x += offset;
		else if (axis == 'y') y += offset;
		else if (axis == 'z') z += offset;

		if (!(x >= 0 && x < BASE_RESOLUTION &&
			y >= 0 && y < BASE_RESOLUTION &&
			z >= 0 && z < BASE_RESOLUTION))
			continue;

		unsigned int neighbor_idx = coords_to_voxel_idx(x, y, z, BASE_RESOLUTION);
		result = result || (d_grid[neighbor_idx] != center_val);
	}

	return result;
}

VOXEL_GRID extract_contour(VOXEL_GRID& d_grid) {

	VOXEL_GRID contour(d_grid.size(), false);


	for (int x = 0; x < BASE_RESOLUTION; x++) {
		for (int y = 0; y < BASE_RESOLUTION; y++) {
			for (int z = 0; z < BASE_RESOLUTION; z++) {
				unsigned int center_idx = coords_to_voxel_idx(x, y, z, BASE_RESOLUTION);
				bool center_val = d_grid[center_idx];
				bool result =
					check_neighbor(d_grid, 'z', x, y, z) ||
					check_neighbor(d_grid, 'y', x, y, z) ||
					check_neighbor(d_grid, 'x', x, y, z);

				contour[center_idx] = !center_val && result;
			}
		}
	}

	return contour;
}

float get_shortest_dist(CGAL::Bbox_3& a, CGAL::Bbox_3& b) {
	// Start by calculating the distance on each axis (x, y, and z)
	float distX = 0.0f;
	if (a.xmax() < b.xmin()) {
		distX = b.xmin() - a.xmax();  // AABB a is to the left of AABB b
	}
	else if (a.xmin() > b.xmax()) {
		distX = a.xmin() - b.xmax();  // AABB a is to the right of AABB b
	}

	float distY = 0.0f;
	if (a.ymax() < b.ymin()) {
		distY = b.ymin() - a.ymax();  // AABB a is below AABB b
	}
	else if (a.ymin() > b.ymax()) {
		distY = a.ymin() - b.ymax();  // AABB a is above AABB b
	}

	float distZ = 0.0f;
	if (a.zmax() < b.zmin()) {
		distZ = b.zmin() - a.zmax();  // AABB a is in front of AABB b
	}
	else if (a.zmin() > b.zmax()) {
		distZ = a.zmin() - b.zmax();  // AABB a is behind AABB b
	}

	// The total shortest distance is the Euclidean distance between the gaps on each axis
	return std::sqrt(distX * distX + distY * distY + distZ * distZ);
}

bool does_overlap_erode(Node cell, CGAL::Bbox_3 p_bbox) {
	int resol = BASE_RESOLUTION / pow(2, cell.level);
	float cell_size = base_cellsize * pow(2, cell.level);

	float scale = 0.6f;
	float base_radius = base_cellsize;
	float radius = base_radius * scale;

	CGAL::Bbox_3 cell_bbox = calc_voxel_bbox(cell.pos, resol, global_min_point, cell_size);
	CGAL::Bbox_3 cell_bbox_pad(
		cell_bbox.xmin() - radius,
		cell_bbox.ymin() - radius,
		cell_bbox.zmin() - radius,
		cell_bbox.xmax() + radius,
		cell_bbox.ymax() + radius,
		cell_bbox.zmax() + radius
	);

	if (p_bbox.xmax() >= cell_bbox_pad.xmin() &&
		p_bbox.ymax() >= cell_bbox_pad.ymin() &&
		p_bbox.zmax() >= cell_bbox_pad.zmin() &&
		p_bbox.xmin() <= cell_bbox_pad.xmax() &&
		p_bbox.ymin() <= cell_bbox_pad.ymax() &&
		p_bbox.zmin() <= cell_bbox_pad.zmax()
		) {

		float shortest_dist = get_shortest_dist(p_bbox, cell_bbox);

		if (shortest_dist > radius) {
			return false;
		}
		else {
			return true;
		}
	}
	return false;
}

VOXEL_GRID execute_erosion(MIPMAP_TYPE& contour_mipmap, VOXEL_GRID& d_grid, VOXEL_GRID& voxel_grid) {

	VOXEL_GRID e_grid = d_grid;

	int mipmap_depth = contour_mipmap.size();

	std::stack<Node> node_stack;
	std::array<unsigned int, 3> num_voxels = { BASE_RESOLUTION, BASE_RESOLUTION, BASE_RESOLUTION };
	for (int x = 0; x < BASE_RESOLUTION; x++) {
		if (x % 5 == 0) std::cout << "erosion x : " << x << "\n";
		for (int y = 0; y < BASE_RESOLUTION; y++) {
			for (int z = 0; z < BASE_RESOLUTION; z++) {
				int flat_idx = coords_to_voxel_idx(x, y, z, num_voxels);
				if (e_grid[flat_idx] == false || voxel_grid[flat_idx] == true) continue;
				CGAL::Bbox_3 p_bbox = calc_voxel_bbox(flat_idx, BASE_RESOLUTION, global_min_point, base_cellsize);
				//SE se;
				//Node current_point = { 0, flat_idx };
				////define_se(current_point, base_cellsize, se, false);
				//define_se(current_point, base_cellsize * 0.6, se, true);

				node_stack.push({ mipmap_depth - 1, 0 });
				while (!node_stack.empty() && e_grid[flat_idx] == true) {
					auto top_node = node_stack.top();
					node_stack.pop();

					if (top_node.level == 0) {
						e_grid[flat_idx] = false;
						//d_grid[top_node.pos] = true;
						std::stack<Node> empty_stack;
						node_stack.swap(empty_stack);
						break;
					}
					else {
						std::vector<Node> subcells = find_subcells(top_node, contour_mipmap);
						for (auto& subcell : subcells) {
							bool subcell_val = contour_mipmap[subcell.level][subcell.pos];
							bool overlap = does_overlap_erode(subcell, p_bbox);
							if (subcell_val == true && overlap) {
								node_stack.push(subcell);
							}
						}
					}
				}
			}
		}
	}
	return e_grid;
}

// This main function loads already saved mipmap and execute closing
//int main(int argc, char* argv[])
//{
//	MIPMAP_TYPE loaded_mipmap;
//	loadBoolVector2D(loaded_mipmap, "C:/Users/jinjo/Documents/TUD/a2024wise/VC_lab/CageModeler/models/mipmap_data/chessBishop_mipmap.bin", base_cellsize, global_min_point);
//	/*bool is_same = areIdentical(mipmap_pyramid, loaded_mipmap);
//	std::cout << "saved base_cellsize: " << base_cellsize << ", loaded: " << base_cellsize2 << "\nare identical: " << is_same;
//	*/
//	VOXEL_GRID d_grid = executeDilation(loaded_mipmap);
//	std::cout << "dilation done. start drawing\n";
//	std::string obj = "chessBishop";
//	draw_voxel(-1, d_grid, BASE_RESOLUTION, obj);
//
//	std::cout << "drawing done. start contour extraction\n";
//	VOXEL_GRID contour = extract_contour(d_grid);
//	draw_voxel(-2, contour, BASE_RESOLUTION, obj);
//
//	MIPMAP_TYPE contour_pyramid = generate_mipmap(contour);
//
//	VOXEL_GRID e_grid = execute_erosion(contour_pyramid, d_grid, loaded_mipmap[0]);
//	draw_voxel(-3, e_grid, BASE_RESOLUTION, obj);
//	return 0;
//}

// This function only executes the voxelization and save the mipmap as bin file.
int main(int argc, char* argv[])
{
	generate_save("chessBishop");
	return 0;
	
}