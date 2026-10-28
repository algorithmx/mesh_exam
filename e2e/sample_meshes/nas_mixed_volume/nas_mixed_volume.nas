$ NASTRAN mixed volumetric fixture: CTETRA + CHEXA(20-node form,
$ midside nodes retained -> hex20) + CPENTA(6-node -> prism6).
BEGIN BULK
GRID,1,0,0.0,0.0,0.0
GRID,2,0,1.0,0.0,0.0
GRID,3,0,0.0,1.0,0.0
GRID,4,0,0.0,0.0,1.0
CTETRA,101,1,1,2,3,4
$ unit hex at x=5..6 (G1-G4 bottom, G5-G8 top)
GRID,5,0,5.0,0.0,0.0
GRID,6,0,6.0,0.0,0.0
GRID,7,0,6.0,1.0,0.0
GRID,8,0,5.0,1.0,0.0
GRID,9,0,5.0,0.0,1.0
GRID,10,0,6.0,0.0,1.0
GRID,11,0,6.0,1.0,1.0
GRID,12,0,5.0,1.0,1.0
$ hex midside nodes (exercised by the 20-node form, dropped for hex8)
GRID,13,0,5.5,0.0,0.0
GRID,14,0,6.0,0.5,0.0
GRID,15,0,5.5,1.0,0.0
GRID,16,0,5.0,0.5,0.0
GRID,17,0,5.5,0.0,1.0
GRID,18,0,6.0,0.5,1.0
GRID,19,0,5.5,1.0,1.0
GRID,20,0,5.0,0.5,1.0
GRID,21,0,5.0,0.0,0.5
GRID,22,0,6.0,0.0,0.5
GRID,23,0,6.0,1.0,0.5
GRID,24,0,5.0,1.0,0.5
CHEXA,102,1,5,6,7,8,9,10,+H1
+H1,11,12,13,14,15,16,17,18,+H2
+H2,19,20,21,22,23,24
$ unit wedge at x=10..11 (G1-G3 bottom, G4-G6 top)
GRID,25,0,10.0,0.0,0.0
GRID,26,0,11.0,0.0,0.0
GRID,27,0,10.5,1.0,0.0
GRID,28,0,10.0,0.0,1.0
GRID,29,0,11.0,0.0,1.0
GRID,30,0,10.5,1.0,1.0
CPENTA,103,1,25,26,27,28,29,+P1
+P1,30
ENDDATA
