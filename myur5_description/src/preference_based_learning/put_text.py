import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/figs/example.png')


a = r'\xi^B'
b = r'\xi^A'


tr1_x = np.array([546, 554, 571, 904, 1266, 1634, 1953, 1937])
tr1_y = [1852, 1778, 1623, 1484, 1443, 1475, 1645, 1811]



tr2_x = np.array([546, 554, 669, 759, 914, 1291, 1716, 1904, 1953, 1937])
tr2_y = [1852, 1778, 1312, 1140, 1017, 1050, 1279, 1459, 1645, 1811]


plt.imshow(img)
#plt.plot(tr1_x, tr1_y, 'o-', color='red', linewidth=1.5)
#plt.plot(tr2_x, tr2_y, 'o-', color='blue', linewidth=1.5)
#plt.arrow(378, 341, 11, 80, color='red', linewidth=0.8, head_length=25, width=3.5)
#plt.arrow(299, 186, 10, 230, color='blue', linewidth=0.8, head_length=25, width=3.5)


#plt.text(1094, 1850, '$%s$' %a, size=25, color='red') 
#plt.text(1242, 900, '$%s$' %b, size=25, color='blue') 



plt.text(232, 80, '$%s$' %a, size=16, color='blue') 
plt.text(313, 80, '$%s$' %b, size=16, color='red') 
plt.axis('off')
plt.savefig('timevarying_example3.png', bbox_inches="tight", pad_inches=0)
plt.show()