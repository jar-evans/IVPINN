// Right Triangles Mesh Generation
lc = 0.9; // Characteristic length

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

Save "_refined0.msh";
RefineMesh;
RefineMesh;
Save "_refined1.msh";
RefineMesh;
RefineMesh;
Save "_refined2.msh";







