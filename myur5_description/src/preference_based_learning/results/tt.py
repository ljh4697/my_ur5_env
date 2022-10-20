from itertools import count
from yaml import DirectiveToken

# 






N, M = map(int, input().split())
x, y, direct = map(int, input().split())

a_MAP = []
visited_place = [[0] * M for i in range(N)]

print(visited_place)

for i in range(N):
    a_MAP.append(list(map(int, input().split())))


visited_place[x][y]=1

result = 1
turntime = 0


dy = [0, 1, 0, -1]; dx = [-1, 0, 1, 0]


def turn_left():
    global direct
    direct -= 1
    if direct == -1:
        direct = 3


while True:    
    turn_left()
    nx = x + dx[direct]; ny = y + dy[direct]
    
    if visited_place[nx][ny]==0 and a_MAP[nx][ny]==0:
        visited_place[nx][ny]=1
        x = nx; y = ny
        result +=1
        turntime = 0
        continue
    else:
        turntime += 1
        
    if turntime==4:
        nx=x-dx[direct]; ny=y-dy[direct]
        if a_MAP[nx][ny]==0:
            x=nx; y=ny
            turntime = 0
        else:
            break
        
    
    
print(result)
    
        
        
        


    