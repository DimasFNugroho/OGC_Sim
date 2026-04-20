"""
OGC Algorithms 1-4 — clean module interface.

Each module provides a self-contained function matching the paper's algorithm:

    algorithm1 — vertexFacetContactDetection  (Sec. 4.1)
    algorithm2 — edgeEdgeContactDetection     (Sec. 4.2)
    algorithm3 — simulationStep               (Sec. 4.3, outer loop)
    algorithm4 — simulationIteration (VBD)    (Sec. 4.3, inner solver)
"""

from ogc_sim.algorithms.algorithm1 import vertex_facet_contact_detection
from ogc_sim.algorithms.algorithm2 import edge_edge_contact_detection
from ogc_sim.algorithms.algorithm3 import simulation_step
from ogc_sim.algorithms.algorithm4 import vbd_iteration
