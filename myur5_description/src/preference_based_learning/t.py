import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker, cm


fname_data = '/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/ctrl_samples/preference_point.csv'
data        = np.genfromtxt(fname_data, delimiter=',')
label   = data[:, 2]
point_data = np.delete(data, 2, axis=1)

number_data = data.shape[0]

point_x = data[:, 0]
point_y = data[:, 1]
label   = data[:, 2]

print('number of data = ', number_data)
print('data type of point x = ', point_x.dtype)
print('data type of point y = ', point_y.dtype)

point_x_class_0 = point_x[label == 0]
point_y_class_0 = point_y[label == 0]

point_x_class_1 = point_x[label == 1]
point_y_class_1 = point_y[label == 1]

# psi data가 point data 라고 가정

point_data = np.delete(data, 2, axis=1)




if __name__ == "__main__":
    print('ee')
    print(np.arange(10)[:9])
    

# plt.figure(figsize=(8,8))   
# plt.title('training data')
# plt.plot(point_x_class_0, point_y_class_0, 'o', color='blue', label='class = 0')
# plt.plot(point_x_class_1, point_y_class_1, 'o', color='red', label='class = 1')
# plt.axis('equal')
# plt.legend()
# plt.tight_layout()
# plt.show()