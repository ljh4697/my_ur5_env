import control_planning_scene
import ur5e_plan




class kinematic_world(object):
    def __init__(self):
        super(kinematic_world, self).__init__()
        self.planning_scene_1 = control_planning_scene.control_planning_scene()
        self.planning_ur5e = ur5e_plan.ur5e_plan()

    
    def pick_and_place(self):
        pass
    
    def kinematic_main(self, env, terminate):
        pass
        
        
        
