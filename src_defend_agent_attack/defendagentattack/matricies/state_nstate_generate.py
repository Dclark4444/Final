import numpy as np
import math 
import os



def helper_valid_act(n_st, c_st, act_ls,
                      who): # if defense is moving bod y, arm has to be zero 
    state_to_act = {0: [0,1,2], 1: [3,4,5]}

    if who == "A" and  n_st[1] > 0 and n_st[0] == 0: 
#for attack, no action allowed to achieve a state where teh arm is not at home base adn the body is home, harms robot
        return None 
    else: 
        

        n_st =np.array(n_st)
        c_st = np.array(c_st)

        diff = n_st - c_st # get the difference for each index 

         # check if you have moved anything  -- no restrictions here 
        tot_mov = sum(abs(diff))

        if tot_mov == 0:  # maybe change this where they can use the same kaction 


            return None 
        
        if 0 in diff: # if at least one part of the body is not moving 
            idx_ch =  int(np.where(diff!= 0)[0])# index for where the change is            
            change_st = int(n_st[idx_ch]) # get the state of the body arf of interest 



# state-to-act[idx_ch] gets teh list of possible actions based on that body part 
# getting change_st idx will allow you to map to the action list of the player 
            act_val = state_to_act[idx_ch][change_st]
        


            return act_val
    
        
        else: 
            return None


def main():

    # Fetch Actions and states (saved files)
    matrix_pah = "."

    action_pth = os.path.join(matrix_pah, "actions.txt")

    actions = np.loadtxt(action_pth)

    actions = np.loadtxt(action_pth)

    state_pth = os.path.join(matrix_pah, "states.txt")
    states = np.loadtxt(state_pth)
    print(states)

    s_len = len(states)
    

    state_nstate = -1*np.ones((s_len,s_len)) # default value is negative one 

    # next state is only possible if both robots perform an action ---> can't stay still
    # with what states are actions avaialbe


        # self.player_exp_actions = {"attack": ["home","M_R","M_L","arm"],
        #                       "defend" : ["home", "A_R", "A_L"]}

    for curr_idx, next_list in enumerate(state_nstate):# for every row index (represents a state )
        # look at each column item ===> get current 
        curr_state = np.array(states[curr_idx])
        for next_idx, val in enumerate(next_list): # for every next_state get the index of stata, and the value ---> set the value at the end of the function 

                    # self.player_exp_actions = {"attack": ["home","M_R","M_L","arm"],
                    #           "defend" : ["home", "A_R", "A_L"]}
            # 
            n_state = np.array(states[next_idx]) # get the next state 

            a_n_st= n_state[0:2]
            a_cur_st= curr_state[0:2]
            d_n_st = n_state[2:4]
            d_curr_s = curr_state[2:4]

            # get the position of the person 

            a_act = helper_valid_act(a_n_st, a_cur_st, actions, "A")          
            d_act = helper_valid_act(d_n_st, d_curr_s, actions, "D")

            # if eitheer of the acts are none --> continue 
            if a_act == None or d_act == None: 
                continue 


            new_act = [a_act, d_act]
            find_act = [i for i, state in enumerate(actions) if state.tolist() == new_act][0]
            # if find_act == 0: 
            #     find_act += 1 

            state_nstate[curr_idx][next_idx] = int(find_act) # or the defalt value 

    file_name = "state_nstate.txt"
    np.savetxt(file_name, state_nstate.astype(int), fmt='%d')

if __name__ == "__main__":
    main() 