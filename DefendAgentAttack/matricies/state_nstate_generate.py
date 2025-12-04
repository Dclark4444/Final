import numpy as np
from ament_index_python.packages import get_package_share_directory

import os

def main():

    # Fetch Actions and states (saved files)
    share = get_package_share_directory('defendagentattack')

    action_pth = os.path.join(share, 'matrices', 'action.txt')
    actions = np.loadtxt(action_pth)

    state_pth = os.path.join(share, 'matrices', 'states.txt')
    states = np.loadtxt(state_pth)

    s_len = len(states)
    

    state_nstate = -1*np.ones((s_len,s_len)) # default value 

    # next state is only possible if both robots perform an action ---> can't stay still
    # with what states are actions avaialbe

    curr_s = states[0]


        # self.player_exp_actions = {"attack": ["home","M_R","M_L","arm"],
        #                       "defend" : ["home", "A_R", "A_L"]}

    for curr_idx, next_list in enumerate(state_nstate):# for every row index (represents a state )
        # look at each column item ===> get current 
        curr_state = np.array(states[curr_idx])
        for next_idx, val in enumerate(next_list):
            # 
            n_state = np.array(states[next_idx])

            diff = n_state - curr_state # calculate the diff [0000]--> 1010 (move_r - A, turn  - D )
            # cases when it doesnt work  0000 -> 

            if sum(diff) <= 1: # both agents need to perform an action
                continue

            a_n_state = n_state[0:1]
            a_cur_state = curr_state[0:1]
            d_n_state = n_state[2:3] 
            d_curr_state = curr_state[2:3]

            # get the position of the person 

            # attack [0,1] --> im facing opnening, and arm donw in direction 
            # attack index 0 has to be > 0 for index 1 to be greater than 0.
            a_arm_bool = a_cur_state[0] == a_n_state[0] and a_cur_state[0] > 0 
            # will be true if you are allowed to move arm down. 
            

            a_diff = a_n_state - a_cur_state
            d_diff = d_n_state - d_curr_state 

            # what ws it before, what is it next ==> what is the action that gets you there

          #  0000 --> 1010 Attack mmove R, Deffend arm_back right --> action [1,1]
          # 1010 --> 1111 Attack arm down  R, move arm 



        # Atack agent states are indicies 0-1 and Defend are 2-3 
        # index 0 represents the Attack Agent stance relative to Defense
        # ==> 0 - in front, 1 - to the Right, 2 - to the left 
        # index 1 represens the A Agent arm up (0), or down (1)
        # index 2 represents the D Agent stance relative to Defense
        # ===> forward facing (0), not seeing (1)
        # index 3 represents the D Agent Arm position 
        # ==> home/up (0), arm_down_right (1), arm_down_left (2)
            

            # what actions get you to the best states 

         # Fetch Actions and states (saved files)
         # Joint action --> [Attack, Defender]
         # Attack Agent can Face/Turn to face opponent (0). Move_R/L (1,2), arm down in direction of move (3)
         # Defend Agent can Turn to Face (0), arm_back_right/left (1,2) 


            # check if both robots actually took an action











    None 
    # do something here to generate the data




if __name__ == "__main__":
    main()