import demos
import sys

#task   = sys.argv[1].lower()
method = sys.argv[1].lower()
N = int(sys.argv[2])
M = int(sys.argv[3])
if method == 'nonbatch' or method == 'random':
    demos.nonbatch(method, N, M)
elif method == 'greedy' or method == 'medoids' or method == 'boundary_medoids' or method == 'successive_elimination':
    b = int(sys.argv[4])
    demos.batch(method, N, M, b)
elif method == 'user_greedy':
    b = int(sys.argv[4])
    demos.user_batch(method, N, M, b)
elif method == "coteaching":
    b = int(sys.argv[4])
    demos.coteaching_batch('greedy', N, M, b)
else:
    print('There is no method called ' + method)

