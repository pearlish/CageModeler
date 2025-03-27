#include<Mesh/Operations/MeshGenerateCageOperation.h>

#include <iostream>
#include <utility>
#include <array>
#include <fstream>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <omp.h>
#include <ctime>
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
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>

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
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_count_stop_predicate.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Convex_hull_3.h>
#include <CGAL/boost/graph/Euler_operations.h>

// save diagnostic state
#pragma GCC diagnostic push 

// turn off the specific warning. Can also use "-Wall"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wmissing-template-arg-list-after-template-kw"


// stuff to define the mesh
#include <vcg/complex/complex.h>

//io
#include <wrap/io_trimesh/import_obj.h>
#include <wrap/io_trimesh/import_off.h>
#include <wrap/io_trimesh/export.h>


// local optimization
#include <vcg/complex/algorithms/local_optimization.h>
#include <vcg/complex/algorithms/local_optimization/tri_edge_collapse_quadric.h>
#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/smooth.h>
//#include <vcg/export_off.h> 




#include <boost/optional/optional_io.hpp>
using namespace vcg;
using namespace tri;
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

typedef CGAL::Simple_cartesian<double>                                  Kernel;
typedef CGAL::Polyhedron_3<Kernel>                                      Surface_mesh;
typedef Kernel::Point_3                                                 Surface_Point;
typedef Kernel::Vector_3                                                 Surface_Vector;
typedef CGAL::Surface_mesh<Surface_Point>                                SurMesh;
typedef boost::graph_traits<SurMesh>::face_descriptor                    FaceIndex;
typedef boost::graph_traits<SurMesh>::halfedge_descriptor                HalfedgeIndex;
typedef boost::graph_traits<SurMesh>::vertex_descriptor                  VertexIndex;


typedef SMS::GarlandHeckbert_plane_policies<Surface_mesh,Kernel>         QEM_Policies;
typedef Surface_mesh::Halfedge_handle                             HalfedgeHandle;
typedef Surface_mesh::Edge_iterator                                EdgeIterator;
typedef CGAL::Side_of_triangle_mesh<Surface_mesh, Kernel>          Point_inside_simp;
typedef boost::graph_traits<Surface_mesh>::edge_descriptor        sedge_descriptor;
typedef boost::graph_traits<Surface_mesh>::halfedge_descriptor    shalfedge_descriptor;


using Grid3D=std::vector<std::vector<std::vector<bool>>>; //represent the voxelized grid
using MipmapTree=std::vector<std::map<std::array<int,3>,bool>>;  //used to save the hierarchy


class MyVertex;
class MyEdge;
class MyFace;

struct MyUsedTypes : public UsedTypes<Use<MyVertex>::AsVertexType, Use<MyEdge>::AsEdgeType, Use<MyFace>::AsFaceType> {};

class MyVertex : public vcg::Vertex< MyUsedTypes,
	vertex::VFAdj,
	vertex::Coord3f,
	vertex::Normal3f,
	vertex::Mark,
	vertex::BitFlags  > {
public:
	vcg::math::Quadric<double>& Qd() { return q; }
	bool is_feature() { return is_feature_; }
	void set_is_feature(bool is_feature) { is_feature_ = is_feature; }
private:
	math::Quadric<double> q;
	bool is_feature_;
};

class MyEdge : public Edge< MyUsedTypes> {};

typedef BasicVertexPair<MyVertex> VertexPair;

class MyFace : public vcg::Face< MyUsedTypes,
	face::VFAdj,
	face::VertexRef,
	face::BitFlags > {};

class MyMesh : public vcg::tri::TriMesh<std::vector<MyVertex>, std::vector<MyFace> > {};
typedef typename MyMesh::ScalarType ScalarType;
typedef typename MyMesh::CoordType CoordType;
typedef MyMesh::VertexType::EdgeType EdgeType;
typedef typename MyMesh::VertexIterator VertexIterator;
typedef typename MyMesh::VertexPointer VertexPointer;
typedef typename MyMesh::FaceIterator FaceIterator;
typedef typename MyMesh::FacePointer FacePointer;

class MyTriEdgeCollapse : public vcg::tri::TriEdgeCollapseQuadric< MyMesh, VertexPair, MyTriEdgeCollapse, QInfoStandard<MyVertex>  > {
public:
	typedef  vcg::tri::TriEdgeCollapseQuadric< MyMesh, VertexPair, MyTriEdgeCollapse, QInfoStandard<MyVertex>  > TECQ;
	typedef  MyMesh::VertexType::EdgeType EdgeType;
	inline MyTriEdgeCollapse(const VertexPair& p, int i, BaseParameterClass* pp) :TECQ(p, i, pp) {}
};




Grid3D initializeGrid(int x, int y, int z){  //initialize the grid with given dimensions in x,y,z drtections

    return Grid3D(x,std::vector<std::vector<bool>>(y,std::vector<bool>(z,false)));
}


void convert_voxel_idx_to_coords(unsigned int idx, std::array<unsigned int, 3> numVoxels, unsigned int& x_idx, unsigned int& y_idx, unsigned int& z_idx)
{
	x_idx = idx / (numVoxels[1] * numVoxels[2]);
	auto const w = idx % (numVoxels[1] * numVoxels[2]);
	y_idx = w / numVoxels[2];
	z_idx = w % numVoxels[2];
}

std::array<ExactPoint, 8> calc_voxel_points(unsigned int idx, std::array<unsigned int, 3> numVoxels, ExactPoint min_point,
	const std::array<ExactVector, 3>& voxel_strides, bool* new_scanline = nullptr)
{
	unsigned int x_idx, y_idx, z_idx;

	convert_voxel_idx_to_coords(idx, numVoxels, x_idx, y_idx, z_idx);

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

	CGAL::make_hexahedron(p[0], p[1], p[3], p[2], p[6], p[4], p[5], p[7], voxel);
	assert(voxel.is_valid());
}

//output the grid into off file
void writeGridIntoObj(const Grid3D& grid, const std::string& filepath){
	std::ofstream file(filepath);
	if(!file.is_open()){
      std::cerr<<"Error:Unable to open the file"<<std::endl;
	  return;
	}

    std::vector<std::array<double,3>> vertices;
    std::vector<std::array<int,4>> faces;

	//traverse the grid, calculate vertices and faces
	for(int x=0;x<grid.size();++x){
		for(int y=0;y<grid[0].size();++y){
			for(int z=0;z<grid[0][0].size();++z){
				if(grid[x][y][z]){
					//calculate the position of 8 vertices of the voxel cube
					int baseIndex=vertices.size()+1;
					double dx=static_cast<double>(x);
					double dy=static_cast<double>(y);
					double dz=static_cast<double>(z);

                    //add 8 vertices of the cube
					vertices.push_back({dx,dy,dz});
					vertices.push_back({dx+1,dy,dz});
					vertices.push_back({dx+1,dy+1,dz});
					vertices.push_back({dx,dy+1,dz});
					vertices.push_back({dx,dy,dz+1});
					vertices.push_back({dx+1,dy,dz+1});
					vertices.push_back({dx+1,dy+1,dz+1});
					vertices.push_back({dx,dy+1,dz+1});

					//add 6 faces of the cube
					faces.push_back({baseIndex,baseIndex+1,baseIndex+2,baseIndex+3});
					faces.push_back({baseIndex+4,baseIndex+5,baseIndex+6,baseIndex+7});
					faces.push_back({baseIndex,baseIndex+1,baseIndex+5,baseIndex+4});
					faces.push_back({baseIndex+1,baseIndex+2,baseIndex+6,baseIndex+5});
					faces.push_back({baseIndex+2,baseIndex+3,baseIndex+7,baseIndex+6});
					faces.push_back({baseIndex+3,baseIndex,baseIndex+4,baseIndex+7});




				}
			}
		}
	}

//write into the obj file

//write the vertices
for(const auto& vertex: vertices){
	file<<"v "<<vertex[0]<<" "<<vertex[1]<<" "<<vertex[2]<<"\n";
}
    
//write the faces
for(const auto& face: faces){
	file<<"f "<<face[0]<<" "<<face[1]<<" "<<face[2]<<" "<<face[3]<<"\n";
}

file.close();
std::cout<<"3D grid has been written to file!"<<std::endl;

}

//build hierarchy from the finest level and save into the mipmapTree
MipmapTree buildHeirarchy(const Grid3D& fineGrid, int levels){  
	MipmapTree mipmaptree;
	Grid3D currentGrid=fineGrid;

	for(int level=0; level<levels; ++level){
	
	//create a map for the current level
	std::map<std::array<int,3>,bool> levelMap;

	//assign the boolean value for the current level
	for(int x=0;x<currentGrid.size(); ++x){
		for(int y=0;y<currentGrid[x].size();++y){
			for(int z=0;z<currentGrid[x][y].size(); ++z){
				if(currentGrid[x][y][z]==true){
					levelMap[{x,y,z}]=true;    //save the occupied voxel
				}
			}
		}
	}

	mipmaptree.push_back(levelMap);   //add into the mipmaptree

	//build the next coarser level
	if(level<levels-1){
		//compute the dimensions of the new level
		int newX=std::max(1,static_cast<int>(currentGrid.size()/2));
		int newY=std::max(1,static_cast<int>(currentGrid[0].size()/2));
		int newZ=std::max(1,static_cast<int>(currentGrid[0][0].size()/2));

        Grid3D nextGrid=initializeGrid(newX,newY,newZ);  //voxel grid of the new level

		//traverse every voxels in the new level
		for(int x=0;x<newX;++x){
			for(int y=0;y<newY;++y){
				for(int z=0;z<newZ;++z){
					bool occupied=false;    //initialized the boolean value of every voxel
					//calculate the offsets when use 8 finer grid to represent 1 coarser grid,the values of dx,dy,dz will iterate between 0,1
					for(int dx=0;dx<2;++dx){
						for(int dy=0;dy<2;++dy){
							for(int dz=0;dz<2;++dz){
								//calculate the 8(2*2*2) finer grids' positions which represent a coarser grid 
								int fineX=2*x+dx;
								int fineY=2*y+dy;
								int fineZ=2*z+dz;

								if(fineX<currentGrid.size() && fineY<currentGrid[0].size() && fineZ<currentGrid[0][0].size()){
									occupied=occupied || currentGrid[fineX][fineY][fineZ];    //calculate the occupation of the coarser grid, it is considered occupied as long as one of the finer grid is occupied
								}
							}
						}
					} 

					nextGrid[x][y][z]=occupied;

				}
			}
		}

		currentGrid=nextGrid;  //downscale recursively
	}
	}

	return mipmaptree;
}


//print the mipmap for debug
void printMipmap(const MipmapTree& mipmaptree){
	for(size_t level=0; level<mipmaptree.size();++level){
		std::cout<<"Level"<<level<<":\n";
		for(const auto& [coord,occupied]: mipmaptree[level]){
			if(occupied){
				std::cout<<"("<<coord[0]<<","<<coord[1]<<","<<coord[2]<<"):"<<occupied<<std::endl;
			}
		}
		std::cout<<"\n";
	}
}


//dilation
//check the intersection between the SE of a voxel and the node

bool checkIntersect(const std::array<int,3>& node, int level,const std::array<int,3>& center,int seSize){
	
	//the region of the node box
	int nodeBoxSize=1<<level;
	int nodeBoxMinX=node[0] * nodeBoxSize, nodeBoxMaxX=(node[0]+1) * nodeBoxSize;
	int nodeBoxMinY=node[1] * nodeBoxSize, nodeBoxMaxY=(node[1]+1) * nodeBoxSize;
	int nodeBoxMinZ=node[2] * nodeBoxSize, nodeBoxMaxZ=(node[2]+1) * nodeBoxSize;


	//the region of the SE
	int seMinX=center[0]-seSize, seMaxX=center[0]+seSize+1;
	int seMinY=center[1]-seSize, seMaxY=center[1]+seSize+1;
	int seMinZ=center[2]-seSize, seMaxZ=center[2]+seSize+1;

	//check the overlap
	bool xOverlap=std::max(nodeBoxMinX,seMinX)< std::min(nodeBoxMaxX,seMaxX);
	bool yOverlap=std::max(nodeBoxMinY,seMinY)< std::min(nodeBoxMaxY,seMaxY);
	bool zOverlap=std::max(nodeBoxMinZ,seMinZ)< std::min(nodeBoxMaxZ,seMaxZ);

return xOverlap && yOverlap && zOverlap;


}

bool isValid(int x, int y,int z,const std::array<int,3>& gridSizes ){
	return x>=0 && x<gridSizes[0] && y>=0 && y< gridSizes[1] && z>=0 && z<gridSizes[2];
}



Grid3D parallelDilationA(Grid3D& grid, MipmapTree& mipmap, int se_scale){
	Grid3D result = grid; 
    int maxLevel = mipmap.size() - 1; 
    int gridSizeX = grid.size();
    int gridSizeY = grid[0].size();
    int gridSizeZ = grid[0][0].size();
	std::vector<std::array<int,3>> originalV;

	for(int x=0;x<gridSizeX;++x){
		for(int y=0;y<gridSizeY;++y){
			for(int z=0;z<gridSizeZ;++z){
				if(result[x][y][z]==false){
					originalV.push_back({x,y,z});
				}
			}
		}
	}

   struct Pair{
	int level;
	std::array<int,3> position;
   };
 
  for(auto ov:originalV){
     
	 std::vector<Pair> stack;
      stack.push_back({maxLevel, {0,0,0}});
	  while(!stack.empty() && result[ov[0]][ov[1]][ov[2]]!=true){
		Pair top=stack.back();
         stack.pop_back();
		 if(top.level==0){
   
			result[ov[0]][ov[1]][ov[2]]=true;
			
		 }
		 else{
			for(int dx=0;dx<=1;++dx){
				for(int dy=0; dy<=1;++dy){
					for(int dz=0;dz<=1;++dz){
						 std::array<int, 3> subNode = {
                                           top.position[0]*2 + dx ,
                                            top.position[1]*2 + dy ,
                                            top.position[2] *2+ dz 
											};

                       if(mipmap[top.level - 1].count(subNode) &&
                         mipmap[top.level - 1].at(subNode) ){
				  
				  if( checkIntersect(subNode, top.level - 1, ov, se_scale)){
                           stack.push_back({top.level-1,subNode});
					        }
						 }
                    
					}
				}
			}
		 }
	  }
  }

return result;

}


//extract 6-connected contour of dilatioin
Grid3D extract_contour(Grid3D& dilation){
	Grid3D result=dilation;
	int gridSizeX=dilation.size();
	int gridSizeY=dilation[0].size();
	int gridSizeZ=dilation[0][0].size();

  	std::vector<std::array<int,3>> originalV;

    //collect the voxels
	for(int x=1;x<gridSizeX-1;++x){
		for(int y=1;y<gridSizeY-1;++y){
			for(int z=1;z<gridSizeZ-1;++z){
				if(result[x][y][z]==true){
					originalV.push_back({x,y,z});
				}
			}
		}
	}

   for(auto ov : originalV){
     
	 //check  6 neighboors of the voxel
	   if(dilation[ov[0]+1][ov[1]][ov[2]]==true &&
	      dilation[ov[0]-1][ov[1]][ov[2]]==true &&
	      dilation[ov[0]][ov[1]+1][ov[2]]==true &&
	      dilation[ov[0]][ov[1]-1][ov[2]]==true &&
	      dilation[ov[0]][ov[1]][ov[2]+1]==true &&
	      dilation[ov[0]][ov[1]][ov[2]-1]==true ){
			result[ov[0]][ov[1]][ov[2]]=false;
		  }
   }

   return result;


}

Grid3D cut_the_half(Grid3D& grid){

    Grid3D result=grid;
	int gridSizeX=grid.size();
	int gridSizeY=grid[0].size();
	int gridSizeZ=grid[0][0].size();

	for(int x=0; x<gridSizeX; ++x){
		for(int y=0; y<gridSizeY; ++y){
			for(int z=gridSizeZ/2;z<gridSizeZ; ++z){
				result[x][y][z]=false;
			}
		}
	}

return result;

}



//create sphere SE
std::vector<std::array<int,3>> generate_sphere_offsets(int radius){
std::vector<std::array<int,3>> offsets;

for(int x=-radius; x<=radius; ++x){
	for(int y=-radius; y<=radius; ++y){
		for(int z=-radius; z<=radius; ++z){
			if((x*x+y*y+z*z)<=std::pow(radius,2)){
				offsets.push_back({x,y,z});
			}
		}
	}
}
return offsets;
	
}


int getIndex(int x, int y, int z, const std::array<int, 3>& gridSizes) {
    return x + y * gridSizes[0] + z * gridSizes[0] * gridSizes[1];
}

bool check_intersect_erosionE(const std::array<int,3>& node, int level,const std::array<int,3>& voxelp,int seSize,const std::array<int,3>& gridSizes,std::vector<std::array<int,3>>& offsets){


bool result=false;
//the rigoin of dilated subnode box
//the rigion of the subnode box
int nodeBoxSize=1<<level;
int nodeBoxMinX=node[0] * nodeBoxSize, nodeBoxMaxX=(node[0]+1) * nodeBoxSize;
int nodeBoxMinY=node[1] * nodeBoxSize, nodeBoxMaxY=(node[1]+1) * nodeBoxSize;
int nodeBoxMinZ=node[2] * nodeBoxSize, nodeBoxMaxZ=(node[2]+1) * nodeBoxSize;

//the rigon of dilated subnode box
std::vector<uint8_t> dilated_voxels(gridSizes[0] * gridSizes[1] * gridSizes[2], 0);

//do the dilation
#pragma omp parallel for collapse(3) 
for(int x=nodeBoxMinX; x<nodeBoxMaxX; ++x){
	for(int y=nodeBoxMinY; y<nodeBoxMaxY; ++y){
		for(int z=nodeBoxMinZ; z<nodeBoxMaxZ; ++z){
           std::array<int,3> box_voxel={x,y,z};
           for(const auto& offset: offsets){
               int nx=x+offset[0];
               int ny=y+offset[1];
               int nz=z+offset[2];
			   if(isValid(nx,ny,nz,gridSizes)){
				  dilated_voxels[getIndex(nx, ny, nz, gridSizes)] = 1;
			   }

		   }

		  
		}
	}
}


//calculate the rigion of checked voxel box
for(int x=voxelp[0];x<=voxelp[0]+1; ++x){
	for(int y=voxelp[1]; y<=voxelp[1]+1; ++y){
		for(int z=voxelp[2]; z<=voxelp[2]+1; ++z){
			std::array<int,3> pos={x,y,z};
			
		  int idx=getIndex(x, y, z, gridSizes);
		  if (dilated_voxels[idx]) {
                    result= true;
					break;
                }
		    
		}
	}
}

return result;

}





//test erosion
struct Pair {
    int level;
    std::array<int, 3> position;
};

Grid3D spaciallyErosionT(Grid3D& dilation, Grid3D& grid, MipmapTree& mipmap, int seScale,std::vector<std::array<int,3>>& offsets) {
    Grid3D result = dilation;
    int maxLevel = mipmap.size() - 1;
    int gridSizeX = grid.size();
    int gridSizeY = grid[0].size();
    int gridSizeZ = grid[0][0].size();
    std::vector<std::array<int, 3>> originalV;

    
    #pragma omp parallel
    {
        std::vector<std::array<int, 3>> local_originalV;
        #pragma omp for collapse(3) nowait
        for (int x = 0; x < gridSizeX; ++x) {
            for (int y = 0; y < gridSizeY; ++y) {
                for (int z = 0; z < gridSizeZ; ++z) {
                    if (result[x][y][z] == true && grid[x][y][z] == false) {
                        local_originalV.push_back({x, y, z});
                    }
                }
            }
        }
        #pragma omp critical
        originalV.insert(originalV.end(), local_originalV.begin(), local_originalV.end());
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < originalV.size(); ++i) {
        auto ov = originalV[i];
        std::deque<Pair> stack;
        stack.push_back({maxLevel, {0, 0, 0}});

        while (!stack.empty() && result[ov[0]][ov[1]][ov[2]] == true) {
            Pair top = stack.back();
            stack.pop_back();

            if (top.level == 0) {
                result[ov[0]][ov[1]][ov[2]] = false;
            } else {

                for (int dx = 0; dx <= 1; ++dx) {
                    for (int dy = 0; dy <= 1; ++dy) {
                        for (int dz = 0; dz <= 1; ++dz) {
                            std::array<int, 3> subNode = {
                                top.position[0] * 2 + dx,
                                top.position[1] * 2 + dy,
                                top.position[2] * 2 + dz
                            };

                            if (mipmap[top.level - 1].count(subNode) &&
                            mipmap[top.level - 1].at(subNode) &&
                            check_intersect_erosionE(subNode, top.level - 1, ov, seScale, {gridSizeX, gridSizeY, gridSizeZ}, offsets)) {
                            stack.push_back({top.level - 1, subNode});
                        }
                        }
                    }
                }
            }
        }
    }

    return result;
}

//meshing


const std::array<std::array<int, 3>, 6> neighbors = {{
    {{-1, 0, 0}}, {{1, 0, 0}},  
    {{0, -1, 0}}, {{0, 1, 0}},  
    {{0, 0, -1}}, {{0, 0, 1}}   
}};

std::array<ExactPoint, 8> compute_voxel_vertices(
    const ExactPoint& origin,
    const std::array<ExactVector, 3>& voxel_strides)
{
    return {origin,
            origin + voxel_strides[0],
            origin + voxel_strides[1],
            origin + voxel_strides[2],
            origin + voxel_strides[0] + voxel_strides[1],
            origin + voxel_strides[0] + voxel_strides[2],
            origin + voxel_strides[1] + voxel_strides[2],
            origin + voxel_strides[0] + voxel_strides[1] + voxel_strides[2]};
}






void add_voxel_faces(
    Exact_Polyhedron& mesh,
    const std::array<ExactPoint, 8>& vertices,
    const std::array<bool, 6>& exposed_faces)
{
  static const std::array<std::array<int, 4>, 6> faces = {{
        {{0, 3, 6, 2}},  // -X
        {{1, 4, 7, 5}},  // +X
        {{1, 5, 3, 0}},  // -Y
        {{2, 6, 7, 4}},  // +Y
        {{1, 0, 2, 4}},  // -Z
        {{3, 5, 7, 6}}   // +Z
    }};

    for (int i = 0; i < 6; ++i) {
        if (exposed_faces[i]) {
           mesh.make_triangle(vertices[faces[i][0]], vertices[faces[i][1]], vertices[faces[i][2]]);
          mesh.make_triangle(vertices[faces[i][2]], vertices[faces[i][3]], vertices[faces[i][0]]);
         
		
		}
    }
}

void extract_surface_from_voxels(
    const Grid3D& grid,
    const std::array<ExactVector, 3>& voxel_strides,
    const ExactPoint& origin,
    const std::string outputfile)
{
	Exact_Polyhedron output_mesh;
    size_t nx = grid.size(), ny = grid[0].size(), nz = grid[0][0].size();
    for (size_t x = 0; x < nx; ++x) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t z = 0; z < nz; ++z) {
                if (!grid[x][y][z]) continue;  

                std::array<ExactPoint, 8> voxel_vertices =
                    compute_voxel_vertices(origin + x * voxel_strides[0] + y * voxel_strides[1] + z * voxel_strides[2], voxel_strides);

                std::array<bool, 6> exposed_faces = {true, true, true, true, true, true};
                for (int i = 0; i < 6; ++i) {
                    int nx = x + neighbors[i][0];
                    int ny = y + neighbors[i][1];
                    int nz = z + neighbors[i][2];
                    if (nx >= 0 && nx < grid.size() &&
                        ny >= 0 && ny < grid[0].size() &&
                        nz >= 0 && nz < grid[0][0].size() &&
                        grid[nx][ny][nz]) {
                        exposed_faces[i] = false;
                    }
                }

                add_voxel_faces(output_mesh, voxel_vertices, exposed_faces);
            }
        }
    }

if (!CGAL::is_closed(output_mesh)) {
    std::cerr << "Warning: Mesh is not closed! Trying to close it..." << std::endl;

    PMP::stitch_borders(output_mesh);
    PMP::triangulate_faces(output_mesh);
    
    if (CGAL::is_closed(output_mesh)) {
        std::cout << "Mesh is now closed after stitching!" << std::endl;
    } else {
        std::cerr << "Error: Mesh is still not closed!" << std::endl;
    }
}

if (!PMP::is_outward_oriented(output_mesh)) {
    PMP::reverse_face_orientations(output_mesh);
    std::cout << "Fixed flipped face orientations!" << std::endl;
}


std::ofstream output(outputfile);
 if(!output){
	std::cerr<<"Error: Cannot open output file"<<outputfile<<std::endl;
	return;
   }

   output<<output_mesh;
   output.close();

   std::cout<<"meshing is done successfully!"<<std::endl;


}



//decimation

void decimation(MyMesh& vcg_mesh, int& smooth, int& FinalSize ){
tri::Smooth<MyMesh>::VertexCoordLaplacianHC(vcg_mesh, smooth);
TriEdgeCollapseQuadricParameter qparams;
	qparams.QualityThr = .3;

	float TargetError = std::numeric_limits<float>::max();
	TargetError = 0.01f;
	qparams.QualityCheck = true;  
	qparams.NormalCheck = true;  
	qparams.OptimalPlacement = true;  
	qparams.ScaleIndependent = true;  
	qparams.PreserveTopology = true;
    

	bool CleaningFlag = true;
	if (CleaningFlag) {
		int dup = tri::Clean<MyMesh>::RemoveDuplicateVertex(vcg_mesh);
		int unref = tri::Clean<MyMesh>::RemoveUnreferencedVertex(vcg_mesh);
		printf("Removed %i duplicate and %i unreferenced vertices from mesh \n", dup, unref);
	}
	//int FinalSize = 400;
	printf("reducing it to %i\n", FinalSize);

	vcg::tri::UpdateBounding<MyMesh>::Box(vcg_mesh);

	// decimator initialization
	vcg::LocalOptimization<MyMesh> DeciSession(vcg_mesh, &qparams);

	int t1 = clock();
	DeciSession.Init<MyTriEdgeCollapse>();
	int t2 = clock();
	//printf("BEFORE: mesh  %d %d \n", vcg_mesh.vn, vcg_mesh.fn);
    //printf("Initial Heap Size %i\n", int(DeciSession.h.size()));

	DeciSession.SetTargetSimplices(FinalSize);
	DeciSession.SetTimeBudget(0.5f);
	DeciSession.SetTargetOperations(100000);
	//if (TargetError < std::numeric_limits<float>::max()) DeciSession.SetTargetMetric(TargetError);

  	while (DeciSession.DoOptimization() && vcg_mesh.fn > FinalSize )
	//	printf("Current Mesh size %7i heap sz %9i err %9g \n", vcg_mesh.fn, int(DeciSession.h.size()), DeciSession.currMetric);

	int t3 = clock();
	if (CleaningFlag) {
		int dup = tri::Clean<MyMesh>::RemoveDuplicateVertex(vcg_mesh);
		int unref = tri::Clean<MyMesh>::RemoveUnreferencedVertex(vcg_mesh);
		int deg_face = tri::Clean<MyMesh>::RemoveDegenerateFace(vcg_mesh);
		int dup_face = tri::Clean<MyMesh>::RemoveDuplicateFace(vcg_mesh);

		//tri::UpdateNormal<MyMesh>::PerVertexPerFace(vcg_mesh);
		//printf("Removed %i duplicate and %i unreferenced vertices from mesh \n", dup, unref);
		//printf("Removed %i duplicate and %i unreferenced faces from mesh \n", dup_face, deg_face);
	}

	//printf("mesh  %d %d Error %g \n", vcg_mesh.vn, vcg_mesh.fn, DeciSession.currMetric);
	//printf("\nCompleted in (%5.3f+%5.3f) sec\n", float(t2 - t1) / CLOCKS_PER_SEC, float(t3 - t2) / CLOCKS_PER_SEC);



}


ExactPoint global_min_point;


void voxelizationA(const std::string& inputfile,
                  Grid3D& grid,
                  const std::string& outputfile,
                  std::array<unsigned int, 3>& numVoxels ,
                  std::array<ExactPoint, 8>& aabb_points,
                  std::array<ExactVector, 3>& voxel_strides,
				  int & se_scale){

std::cout << "Loading surface\n";
    Exact_Polyhedron poly;
    if (!PMP::IO::read_polygon_mesh(inputfile, poly) || !CGAL::is_triangle_mesh(poly))
    {
        std::cerr << "Invalid input.\n";
        return ;
    }
    Mesh surface;
    if (!CGAL::Polygon_mesh_processing::IO::read_polygon_mesh(inputfile, surface) || surface.is_empty())
    {
        std::cerr << "Invalid input file.\n";
        return ;
    }

	std::cout << "Compute AABB\n";
    
    CGAL::Bbox_3 bbox = CGAL::Polygon_mesh_processing::bbox(surface);
    ExactPoint min_point(bbox.xmin(), bbox.ymin(), bbox.zmin());
    ExactPoint max_point(bbox.xmax(), bbox.ymax(), bbox.zmax());

    aabb_points = {
        min_point,
        ExactPoint(max_point.x(), min_point.y(), min_point.z()),
        ExactPoint(max_point.x(), max_point.y(), min_point.z()),
        ExactPoint(min_point.x(), max_point.y(), min_point.z()),
        ExactPoint(min_point.x(), min_point.y(), max_point.z()),
        ExactPoint(max_point.x(), min_point.y(), max_point.z()),
        max_point,
        ExactPoint(min_point.x(), max_point.y(), max_point.z())
    };

    std::array<ExactVector, 3> small_voxel_strides = {
        (aabb_points[1] - aabb_points[0]) / static_cast<double>(numVoxels[0]),
        (aabb_points[3] - aabb_points[0]) / static_cast<double>(numVoxels[1]),
        (aabb_points[4] - aabb_points[0]) / static_cast<double>(numVoxels[2])
    };
     std::cout<<"AABB is done"<<std::endl;
     
    //calculate the extended obb,the offset should be se_scale number of tight voxels
    ExactVector new_stride_x=small_voxel_strides[0]*(numVoxels[0]+8.8)/numVoxels[0];
    ExactVector new_stride_y=small_voxel_strides[1]*(numVoxels[1]+8.8)/numVoxels[1];
    ExactVector new_stride_z=small_voxel_strides[2]*(numVoxels[2]+8.8)/numVoxels[2];

    voxel_strides={new_stride_x,new_stride_y,new_stride_z};
    
    //calculate the new min_point as global min point
	ExactPoint new_min_point= aabb_points[0]-(small_voxel_strides[0]*4.4)
	                                             -(small_voxel_strides[1]*4.4)
												 -(small_voxel_strides[2]*4.4);
	 global_min_point=new_min_point;

	//start voxelization
	auto numVoxel = numVoxels[0] * numVoxels[1] * numVoxels[2];
    bool interior = false;

	 std::vector<uint8_t> voxels_marking(numVoxel, 0); // either outside 0, surface 1 or interior 2
    Tree tree(faces(poly).first, faces(poly).second, poly);
	tree.accelerate_distance_queries();

	Point_inside inside_tester(tree);

    #pragma omp parallel for schedule(dynamic)
     for (unsigned int i = 0; i < numVoxel; ++i){
        Exact_Polyhedron voxel = Exact_Polyhedron();
        bool new_scanline;

        calc_voxel_from_idx_tets(i, numVoxels, new_min_point, voxel_strides, voxel, &new_scanline);
      
       //check if inside
       bool inside = true;

        for (auto vert : voxel.vertex_handles())
        {
            if (inside_tester(vert->point()) != CGAL::ON_BOUNDED_SIDE)
            {
                inside = false;
                break;
            }
        }
     CGAL::Iso_cuboid_3<Exact_Kernel> voxel_bbox = CGAL::bounding_box(voxel.points().begin(), voxel.points().end());
     bool intersects = tree.any_intersected_primitive(voxel_bbox).has_value();  

      if (intersects|| inside){
            #pragma omp critical
            {
            grid[i/(numVoxels[1]*numVoxels[2])][(i/numVoxels[2])%numVoxels[1]][i%numVoxels[2]]=true;
            voxels_marking[i]=inside ? 2 : 1;
           }
      }


     }

	 	 std::cout<<"parallel part is done"<<std::endl;


				  }

//tri_quad cage
double compute_angle_between_faces(const SurMesh& mesh, FaceIndex f1, FaceIndex f2) {
    Surface_Vector n1 = PMP::compute_face_normal(f1, mesh);
    Surface_Vector n2 = PMP::compute_face_normal(f2, mesh);

	 double dot_product = n1 * n2;  
    double angle_rad = std::acos(std::clamp(dot_product, -1.0, 1.0));  
    return CGAL::to_double(angle_rad) * (180.0 / CGAL_PI);  
}


bool find_best_merge(SurMesh& mesh, FaceIndex f, double angle_threshold, std::unordered_set<FaceIndex>& merged_faces) {
    double min_angle = angle_threshold; 
    HalfedgeIndex best_halfedge;
    FaceIndex best_neighbor = SurMesh::null_face();

    for (HalfedgeIndex h : CGAL::halfedges_around_face(halfedge(f, mesh), mesh)) {
        FaceIndex neighbor = CGAL::face(opposite(h, mesh), mesh);
        if (neighbor == SurMesh::null_face() || merged_faces.count(neighbor)) continue; 

        double angle = compute_angle_between_faces(mesh, f, neighbor);
        if (angle < min_angle) {
            min_angle = angle;
            best_halfedge = h;
            best_neighbor = neighbor;
        }
    }

    if (best_neighbor != SurMesh::null_face()) {
        CGAL::Euler::join_face(best_halfedge, mesh); 
        merged_faces.insert(f);
        merged_faces.insert(best_neighbor);
        return true;
    }
    return false;
}




void convert_to_tri_quad_meshT(SurMesh& mesh, double angle_threshold=10.0) {
    std::unordered_set<FaceIndex> merged_faces;
    std::vector<FaceIndex> faces_list(faces(mesh).begin(), faces(mesh).end());

    for (FaceIndex f : faces_list) {
        if (merged_faces.count(f)) continue; 
        find_best_merge(mesh, f, angle_threshold, merged_faces);
    }
}

void GenerateCageFromMeshOperation::Execute(){

 std::string filename = _params._meshfilepath.string();
 std::string outputfilename=_params._cagefilepath.string();

 int smooth=_params._smoothIterations;
 int cageFace=_params._targetNumFaces;
bool isTriQuad=_params._isTriQuad;

//extract input model name
std::string obj=filename.substr(filename.find_last_of('/')+1,filename.find_last_of('.')-1);
std::string filepath=filename.substr(0,filename.find_last_of('/')+1);
std::string intermediate_path=filepath+obj+"_interm.obj";

//voxel number
    std::array<unsigned int, 3> numVoxels = { 64u, 64u, 64u };
 Grid3D fineGrid=initializeGrid(numVoxels[0],numVoxels[1],numVoxels[2]);
 std::array<ExactPoint, 8> obb_points;
std::array<ExactVector, 3> voxel_strides;

const std::string off_filename=filepath+"voxelization.off";
const std::string grid_filename=filepath+"grid.obj";
const std::string dilation_obj_cube_filename=filepath+"dialation_cube.obj";
const std::string erosion_obj_filename=filepath+"erosion.obj";
const std::string meshing_off_filename=filepath+"meshing4.off";
std::string tri_quad_intermediate_path=filepath+"tri_quad_inter.obj";



	int se_scale=4;
	const int hierarchyLevels=7;
   auto start_voxelization = std::chrono::high_resolution_clock::now(); 
   voxelizationA(filename,fineGrid,off_filename,numVoxels,obb_points,voxel_strides,se_scale);
   	auto end_voxelization = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> elapsed_voxelization = end_voxelization - start_voxelization;
    	std::cout<<"voxelization is done, elapse time: "<< elapsed_voxelization.count()<<" second "<<std::endl;

   std::vector<std::array<int,3>> offsets=generate_sphere_offsets(se_scale-3);

   	 //wirte the 3D grid into the obj file
     writeGridIntoObj(fineGrid,grid_filename);

	 //build the mipmaptree for dilation
        auto start_dilation = std::chrono::high_resolution_clock::now(); 

	   MipmapTree mipmap_dilate = buildHeirarchy(fineGrid,hierarchyLevels);

      //do the dilation
	  
	    Grid3D dilation=parallelDilationA(fineGrid,mipmap_dilate,se_scale);
      auto end_dilation = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> elapsed_dilation = end_dilation - start_dilation;
    std::cout<<"dilation is done ,elapse time: "<<elapsed_dilation.count()<<" second "<<std::endl;
 
      writeGridIntoObj(dilation,dilation_obj_cube_filename);

	  	//extract the contour of the dilation
		Grid3D contour=extract_contour(dilation);

		//build the mipmap for eroion
        auto start_erosion = std::chrono::high_resolution_clock::now(); 

		MipmapTree mipmap_erose=buildHeirarchy(contour,hierarchyLevels);

       //do the erosion
		

		Grid3D erosion=spaciallyErosionT(dilation,fineGrid,mipmap_erose,se_scale-1,offsets);
        auto end_erosion = std::chrono::high_resolution_clock::now(); 
        std::chrono::duration<double> elapsed_erosion = end_erosion - start_erosion;

		std::cout<<"erosion is done, elapse time: "<<elapsed_erosion.count()<< "second "<<std::endl;

		//write the erosion(grid) into obj file
		writeGridIntoObj(erosion,erosion_obj_filename);

       //meshing
	    auto start_meshing = std::chrono::high_resolution_clock::now(); 
		extract_surface_from_voxels(erosion,voxel_strides,global_min_point,meshing_off_filename);

       //decimation
		MyMesh final_mesh;
		tri::io::ImporterOFF<MyMesh>::Open(final_mesh,meshing_off_filename.c_str());
		decimation(final_mesh,smooth,cageFace);
      auto end_meshing = std::chrono::high_resolution_clock::now(); 
        std::chrono::duration<double> elapsed_meshing = end_meshing - start_meshing;
       	std::cout<<"meshing is done, elapse time: "<<elapsed_meshing.count()<< "second "<<std::endl;

      if(!isTriQuad){

        tri::io::ExporterOBJ<MyMesh>::Save(final_mesh,outputfilename.c_str(),tri::io::Mask::IOM_BITPOLYGONAL);

      }
      else{
        tri::io::ExporterOBJ<MyMesh>::Save(final_mesh,tri_quad_intermediate_path.c_str(),tri::io::Mask::IOM_BITPOLYGONAL);
         
		   SurMesh input_mesh;
		
    if (!CGAL::Polygon_mesh_processing::IO::read_polygon_mesh(tri_quad_intermediate_path, input_mesh) || input_mesh.is_empty())
    {
        std::cerr << "Invalid input file.\n";
        return ;
    }

	convert_to_tri_quad_meshT(input_mesh);
        std::ofstream out(outputfilename);
    if (!out) {
        std::cerr << "Error: Cannot open output.obj for writing!" << std::endl;
        return ;
    }

    if (!CGAL::IO::write_OBJ(out, input_mesh)) {
        std::cerr << "Error: Failed to write OBJ file!" << std::endl;
        return ;
    }
      }
      
    

        
	

}