import trimesh
import os



dir_path = os.path.dirname(os.path.realpath(__file__))
mesh = trimesh.load(dir_path + '/pedestal.dae', force='mesh')

print(mesh.bounding_box.extents)
print(mesh.bounding_box_oriented.primitive.extents)


mesh.show()
mesh.bounding_box_oriented.show()