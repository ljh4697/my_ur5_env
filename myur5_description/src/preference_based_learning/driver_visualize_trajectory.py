import pyglet
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import gym
import time
import numpy as np
import pyglet.gl as gl
from world import World
import car
import dynamics
import visualize
import lane
import pyglet.graphics as graphics
import math
from pyglet import shapes
import pyglet.window.key as key


window = pyglet.window.Window(600, 600, fullscreen=False, caption='unnamed')


grass = pyglet.resource.texture('imgs/grass.png')
clane = lane.StraightLane([0., -1.], [0., 1.], 0.17)
world = World()
world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
world.roads += [clane]


dyn = dynamics.CarDynamics(0.1)
robot = car.Car(dyn, [0., -0.3, np.pi/2., 0.4], color='orange')
human = car.Car(dyn, [0.17, 0., np.pi/2., 0.41], color='white')


world.cars.append(robot); world.cars.append(human)










def centered_image(filename):
    img = pyglet.resource.image(filename)
    img.anchor_x = img.width/2.
    img.anchor_y = img.height/2.
    return img

def car_sprite(color, scale=0.15/600.):
    sprite = pyglet.sprite.Sprite(centered_image('imgs/car-{}.png'.format(color)), subpixel=True)
    sprite.scale = scale
    return sprite


sprites = {c :car_sprite(c) for c in ['red', 'yellow', 'purple', 'white', 'orange', 'gray', 'blue']}


def draw_lane_surface(lane):
    gl.glColor3f(0.4, 0.4, 0.4)
    W = 1000
    graphics.draw(4, gl.GL_QUAD_STRIP, ('v2f',
        np.hstack([lane.p-lane.m*W-0.5*lane.w*lane.n, lane.p-lane.m*W+0.5*lane.w*lane.n,
                    lane.q+lane.m*W-0.5*lane.w*lane.n, lane.q+lane.m*W+0.5*lane.w*lane.n])
    ))
def draw_lane_lines(lane):
    gl.glColor3f(1., 1., 1.)
    W = 1000
    graphics.draw(4, gl.GL_LINES, ('v2f',
        np.hstack([lane.p-lane.m*W-0.5*lane.w*lane.n, lane.p+lane.m*W-0.5*lane.w*lane.n,
                    lane.p-lane.m*W+0.5*lane.w*lane.n, lane.p+lane.m*W+0.5*lane.w*lane.n])
    ))
    
    
def draw_trajectory():
    gl.glColor3f(1., 1., 1.)
    W = 1000
    tr1 = np.load('./trajectory_ex/driver/tj1_f.npz')
    tr2 = np.load('./trajectory_ex/driver/tj2_f.npz')
    
    
    batch = pyglet.graphics.Batch()
    
    width = 0.012
    size_circle = 0.017
    
    for i in range(0,len(tr1['human'])-3, 3):
    
    # size of circle
    # color = green

        
        globals()['line1'+str(i)] = shapes.Line(tr1['human'][i, 0], tr1['human'][i, 1],
                                tr1['human'][i+3, 0], tr1['human'][i+3, 1], width, color =(160, 20, 20))
        
        globals()['line1'+str(i)].draw()
        globals()['circle1'+str(i)] = shapes.Circle(tr1['human'][i, 0], tr1['human'][i, 1], size_circle, color =(160, 20, 20))
        
        globals()['circle1'+str(i)].draw()
        globals()['line'+str(i)] = shapes.Line(tr1['robot'][i, 0], tr1['robot'][i, 1],
                                tr1['robot'][i+3, 0], tr1['robot'][i+3, 1], width, color =(255, 255, 255))
        
        globals()['line'+str(i)].draw()
        globals()['circle'+str(i)] = shapes.Circle(tr1['robot'][i, 0], tr1['robot'][i, 1], size_circle, color =(255, 255, 255))
        
        
        globals()['circle'+str(i)].draw()
        
        
    for i in range(0,len(tr2['human'])-2, 3):
        
        globals()['line2'+str(i)] = shapes.Line(tr2['human'][i, 0], tr2['human'][i, 1],
                                tr2['human'][i+3, 0], tr2['human'][i+3, 1], width, color =(10, 10, 160), batch=batch)
        
        globals()['line2'+str(i)].draw()
        
        globals()['circle2'+str(i)] = shapes.Circle(tr2['human'][i, 0], tr2['human'][i, 1], size_circle, color =(10, 10, 160), batch=batch)
        
        globals()['circle2'+str(i)].draw()
            

        
        
    
    
def draw_car(x, color='yellow', opacity=255):
    
    
    sprite = sprites[color]
    sprite.x, sprite.y = x.data0['x0'][0], x.data0['x0'][1]
    sprite.rotation = -x.data0['x0'][2]*180./math.pi
    sprite.opacity = opacity
    sprite.draw()


@window.event
def on_key_press(symbol, modifier):
   
    # key "C" get press
    if symbol == key.C:
        pyglet.image.get_buffer_manager().get_color_buffer().save('./results/saved_graph/driver/driver_sample.png')
        # printing the message
        print("Key : C is pressed")


@window.event
def on_draw():
    window.clear()
    
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glPushMatrix()
    gl.glLoadIdentity()
    gl.glEnable(grass.target)
    gl.glEnable(gl.GL_BLEND)
    gl.glBindTexture(grass.target, grass.id)
    W = 10000.
    graphics.draw(4, gl.GL_QUADS,
        ('v2f', (-W, -W, W, -W, W, W, -W, W)),
        ('t2f', (0., 0., W*5., 0., W*5., W*5., 0., W*5.))
    )
    gl.glDisable(grass.target)
    for lane in world.lanes:
        draw_lane_surface(lane)
    for lane in world.lanes:
        draw_lane_lines(lane)
    for car in world.cars:
        draw_car(car, car.color)
        
    draw_trajectory()
    print('*** press \'c\' for screen shot ***')
    

pyglet.app.run()