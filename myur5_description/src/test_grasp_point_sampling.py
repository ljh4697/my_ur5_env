import numpy as np
import trimesh
'''
2021 12 26
trimesh sample 을 이용하여 mesh face 에서 random 하게 vertice 를 샘플 해봤다
mesh vertice = gray, sampled vertice = red, rayed vertice = blue, origin = green

trimesh를 사용하여 sampling 된 points 중에서 center mass 와 가장 가까운 point 를 구해서 rayed point 를 구했다.

sample 된 두 point를 있는 vector 와 0,0,1 과 cross_product 하여 normal 한 vector (approach direction) 을 찾음

위에서 구한 approach direction 을 이용하여 mesh 와 approach direction 방향으로 거리를 두어 catesian path planning 으로 pick and place




future work:

    x 축으로 gripper가 들어갈 것이기 때문에 그 것을 고려하여 face 를 reject 하여 sampling 하자 (approach direction과 평행한 face)
    (좋은 방법이 아님)
    
    kinematic world 에서 pick and place 하는 것 까지 성공, dynamic world 에서 실험해보자
    
'''

def cross_product(x, y):
    z = np.zeros(3)
    z[0] = x[1]*y[2]-x[2]*y[1]
    z[1] = x[2]*y[0]-x[0]*y[2]
    z[2] = x[0]*y[1]-x[1]*y[0]
    

    return z, np.arcsin(np.linalg.norm(z)/(np.linalg.norm(x)*np.linalg.norm(y)))


    
# a = np.array([[1, 2, 3], [2, 2, 4], [3, 4, 5]])
# b = np.array([[4, 4, 5]])

# print(np.concatenate((a, b), axis=0))

# print(np.where(a== (1, 2, 3)))
# print(np.where((a== (1, 2, 3)).all(axis=1)))



mesh = trimesh.load("/home/joonhyeok/robosuite/robosuite/models/assets/objects/meshes/bottle.stl", force='mesh')

mesh_center_mass = mesh.center_mass

mesh_vertices = mesh.vertices
safe_distance = 0.03/0.9

grasp_point = [0, 0, np.max(mesh_vertices[:, 2])-safe_distance]

safe_range = (grasp_point[2]-0.01, grasp_point[2]+0.01)

dot0_face_z = np.where(np.dot(mesh.face_normals[:],np.array([0, 0, 1]))==0)[0]
#dot0_face_x = np.where(np.dot(mesh.face_normals[:],np.array([1, 0, 0]))==0)[0]



# face_processing = []

# for i in range(len(mesh.face_adjacency[:,0])):
#     if mesh.face_adjacency[i,0] in dot0_face and mesh.face_adjacency_angles[i] == 0:
#         face_processing.append(mesh.face_adjacency[i,0])
        
# selected_face = np.array(sorted(set(face_processing)))

# face_processing = []


# for i in range(len(selected_face)):
#     if np.max(mesh.vertices[mesh.faces[selected_face[i], :],2]) <= np.max(safe_range) and np.min(mesh.vertices[mesh.faces[selected_face[i], :],2]) >= np.min(safe_range):
#         face_processing.append(selected_face[i])
#mesh.face_adjacency[:, 1] 

num_faces = len(dot0_face_z)
weights = np.zeros(len(mesh.faces))
weights[dot0_face_z] = 1/num_faces



# point sampling
samples, face_idx = mesh.sample(120, return_index = True, face_weight = weights)
sample_opposite_normals = np.zeros_like(samples)
sample_opposite_normals = -mesh.face_normals[face_idx, :]

# find the closest sample with center mass
mesh_centeres = np.ones_like(samples)*mesh_center_mass
distances = np.linalg.norm(mesh_center_mass-samples, axis=1)
cloeset_point = samples[np.argmin(distances)]
normal_vector = sample_opposite_normals[np.argmin(distances)]
# find rayed point
#location, idx, f = mesh.ray.intersects_location(ray_origins=samples, ray_directions=sample_opposite_normals)
location, idx, f = mesh.ray.intersects_location(ray_origins=np.array([cloeset_point]), ray_directions=np.array([normal_vector]))


approach_direction, _= cross_product(location[1]-location[0], [0, 0, 1])
_, degree = cross_product([1, 0, 0], approach_direction)
approach_point = (location[1]+location[0])/2

print(approach_direction)
print(degree)
print(approach_point)







# test pointcloud with mesh_vertices & sample_vertices
origin = np.array([[0, 0, 0]])
#location = np.array([location[-1]])
cloeset_point = np.array([cloeset_point])

#vertices = np.concatenate((mesh.vertices,samples), axis=0)
vertices = np.concatenate((mesh.vertices,origin), axis= 0)
vertices = np.concatenate((vertices,location), axis= 0)
#vertices = np.concatenate((vertices,cloeset_point), axis= 0)
cloud = trimesh.points.PointCloud(vertices)
initial_md5 = cloud.md5()


# set some colors (mesh_vertices = gray, sample_vertices = red (R, G, B, alpha))
m_clrs =np.ones((len(mesh.vertices), 4))
m_clrs[:, 3] = 0.6
sample_clrs = np.ones((len(samples), 4))
sample_clrs[:, 1] = 0
sample_clrs[:, 2] = 0

origin_clr = np.array([[0, 1, 0, 1]])
closest_point_clr = np.array([[1, 0, 0, 1]])
#grasp_point_clr = np.array([[0, 0, 1, 1]])
location_clrs = np.ones((len(location), 4))
location_clrs[:, 1] = 0
location_clrs[:, 0] = 0

#clrs = np.concatenate((m_clrs, sample_clrs), axis=0)
clrs = np.concatenate((m_clrs, origin_clr), axis=0)
clrs = np.concatenate((clrs, location_clrs), axis=0)
#clrs = np.concatenate((clrs, closest_point_clr), axis=0)
cloud.colors = clrs

# remove the duplicates we create
cloud.merge_vertices()
cloud.show()
cloud.scene()


mesh.show()





































# pointcloud 테스트 !!

# shape = (100, 3)

# points = np.random.random(shape)


# # make sure randomness never gives duplicates by offsetting
# points += np.arange(shape[0]).reshape((-1, 1))


# # make some duplicate vertices
# points[:10] = points[0]

# # create a pointcloud object
# cloud = trimesh.points.PointCloud(points)

# initial_md5 = cloud.md5()


# # set some random colors
# cloud.colors = np.random.random((shape[0], 4))
# # remove the duplicates we created
# cloud.merge_vertices()

# cloud.show()

# # new shape post- merge
# new_shape = (shape[0] - 9, shape[1])
