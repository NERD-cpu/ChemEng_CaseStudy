// Gmsh mesh for single packed bed tube

// Reactor geometry
Point(11) = {0, 0, 0, 1};
Point(12) = {0, 0.0127, 0, 1};
Point(13) = {3.0, 0, 0, 1};
Point(14) = {3.0, 0.0127, 0, 1};
Line(10) = {11, 12};               
Line(20) = {12, 14};           
Line(30) = {14, 13};       
Line(40) = {13, 11};       
Line Loop(100) = {10, 20, 30, 40};

// Physical entities
Plane Surface(1000) = {100};
Physical Surface(0) = {1000};
Physical Curve(1) = {10};
Physical Curve(2) = {20};
Physical Curve(3) = {30};
Physical Curve(4) = {40};
