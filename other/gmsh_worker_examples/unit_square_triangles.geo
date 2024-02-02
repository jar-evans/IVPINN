lc = 1;
// Define the geometry
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc};

// Define the right triangles
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};

// Set mesh size
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

// Set the mesh element type to 2D triangles
Mesh.ElementOrder = 1;
Mesh.ElementType = 2;

// Generate the mesh
Mesh 2;
Save "0_refined.msh";
RefineMesh;
Save "1_refined.msh";
RefineMesh;
Save "2_refined.msh";
RefineMesh;
Save "3_refined.msh";
RefineMesh;
Save "4_refined.msh";
RefineMesh;
Save "5_refined.msh";
RefineMesh;
Save "6_refined.msh";
