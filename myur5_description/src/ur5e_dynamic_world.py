import robosuite as suite
import numpy as np


class dynamic_world(object):
    def __init__(self):
        super(dynamic_world, self).__init__()
        
        
        
        
        
    def dynamic_main(self, env, plan):
        desired_positions = plan
        
        i=0
        delta_q = np.zeros(7)
        goal_reach = False
        terminate = False
        env.viewer.set_camera(camera_id=0)

        while not terminate:
            s = 0
            
            while not goal_reach:
                current_positions = env.robots[0]._joint_positions
                delta_q[:6] = desired_positions[i]-current_positions
                action = delta_q.copy()
                # if action.sum() < 0.01:
                #     action *= 2
                #if s > 250:
                action *= (s/100)
                    
                #check joint position
                #env.robots[0].sim.data.qpos[:6] = desired_positions[i]


                #obs, reward, done, info = env.step(action)  # take action in the environment
                env.step(action)
                env.render()  # render on display



                goal_reach = True
                print(delta_q, 'step', i, 's = ', action.sum(), s)
                for e in range(len(delta_q)):
                    if np.abs(delta_q[e]) > 0.003:
                        goal_reach = False
                        break
                    
                s += 1
                #print(env.robots[0]._hand_pos)


            goal_reach = False

            if i < len(desired_positions)-1:
                i+=1
            else:
                terminate = True
                
        return terminate
                
                
        
        
                
            
