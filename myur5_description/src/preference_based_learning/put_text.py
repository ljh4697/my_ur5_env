import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/timevarying_example.jpg')


a = r'\xi^A'
b = r'\xi^B'


tr1_x = np.array([546, 554, 554, 904, 1225, 1692, 1937])
tr1_y = [1852, 1778, 1484, 1484, 1598, 1553, 1729]



tr2_x = np.array([267, 243, 223, 128, 232, 365, 367, 299])
tr2_y = [436, 266, 164, 71, 19, 38, 138, 186]


plt.imshow(img)
plt.plot(tr1_x, tr1_y, 'o-', color='red', linewidth=1.5)
#plt.plot(tr2_x, tr2_y, 'o-', color='blue', linewidth=1.5)
#plt.arrow(378, 341, 11, 80, color='red', linewidth=0.8, head_length=25, width=3.5)
#plt.arrow(299, 186, 10, 230, color='blue', linewidth=0.8, head_length=25, width=3.5)


plt.text(54, 169, '$%s$' %a, size=25, color='darkblue') 
plt.text(110, 46, '$%s$' %b, size=25, color='red') 
plt.axis('off')
plt.savefig('traj_timevarying_example.png', bbox_inches="tight", pad_inches=0)
plt.show()