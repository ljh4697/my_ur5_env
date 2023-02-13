import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('/home/joonhyeok/catkin_ws/src/my_ur5_env/myur5_description/src/preference_based_learning/figs/userstudy.png')


a = r'\xi^B'
b = r'\xi^A'


tr1_x = np.array([564, 564, 562, 533, 468, 396, 368, 350, 350])
tr1_y = [397, 363, 306, 240, 175,208, 282, 332, 371]



tr2_x = np.array([564, 564, 477, 433, 385, 350, 350])
tr2_y = [397, 363, 347, 334, 326, 332, 371]


plt.imshow(img)
plt.plot(tr1_x, tr1_y, 'o-', color='blue', linewidth=1.5)
plt.plot(tr2_x, tr2_y, 'o-', color='red', linewidth=1.5)
#plt.arrow(378, 341, 11, 80, color='red', linewidth=0.8, head_length=25, width=3.5)
#plt.arrow(299, 186, 10, 230, color='blue', linewidth=0.8, head_length=25, width=3.5)


#plt.text(1094, 1850, '$%s$' %a, size=25, color='red') 
#plt.text(1242, 900, '$%s$' %b, size=25, color='blue') 



plt.text(260, 334, '$%s$' %b, size=24, color='red') 
plt.text(558, 166, '$%s$' %a, size=24, color='blue') 
plt.axis('off')
plt.savefig('userstudy_fig.png', bbox_inches="tight", pad_inches=0)
plt.show()