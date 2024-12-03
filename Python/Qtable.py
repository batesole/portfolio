#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 18:41:35 2022

Using the OpenAI Gym Frozen Lake Environment, find the best policy for a 
Q table using the provided deterministic and non-deterministic (respectively) 
formulas:
    
Q(S, A) ← R + γ*maxa Q(S' , a)
Q(S, A) ← Q(S, A) + α(R + γ*maxa Q(S', a) − Q(S, A))
   γ = 0.9
   R = 1.0 at state 16
   α = 0.5
   
Do experiments for deterministic cases with deterministic update rules, as well
as non-deterministic cases with both deterministic and non-deterministic update
rules.  Non-deterministic cases use the is_slippery=True parameter, which is 
false in deterministic cases.                         

@author: Ash Batesole
"""

import gym
import time
start_time = time.time()
import numpy as np
import matplotlib.pyplot as plt

# evaluate a q table
def eval_policy(qtable_, num_of_episodes_, max_steps_):
    rewards = []

    for episode in range(num_of_episodes_):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0

        for step in range(max_steps_):
            # if there is no max in the current state, choose a random action
            if np.amax(qtable_[state,:])==0:
                action = np.random.randint(0,4)
            else:   
                action = np.argmax(qtable_[state,:])
            # action = np.argmax(qtable_[state,:])
            new_state, reward, done, info = env.step(action)
            total_rewards += reward
        
            if done:
                rewards.append(total_rewards)
                break
            state = new_state
    env.close()
    avg_reward = sum(rewards)/num_of_episodes_
    return avg_reward




# load the Frozen Lake environment and render it
# we will start with a deterministic environment
env = gym.make("FrozenLake-v1", is_slippery=False)
env.reset()
# env.render()

# number of actions and states in the environment
action_size = env.action_space.n
state_size = env.observation_space.n


# code below is used to get familiar with the environment
# # perform actions until an end result occurs 
# done = False
# env.reset()
# while not done:
#     # input random actions
#     # action = np.random.randint(0,4) # 0:Left 1:Down 2: Right, 3: Up
#     # allow the user to choose the actions
#     action = int(input('0/left 1/down 2/right 3/up:'))
#     new_state, reward, done, info = env.step(action)
#     time.sleep(1.0) 
#     print(f'S_t+1={new_state}, R_t+1={reward}, done={done}')
#     env.render()




# number of times we want to run each training
n = 10


# -------deterministic case with deterministic update rule

# select number of episodes and max steps
# max steps are set to make sure program doesn't get stuck in a loop
total_episodes = 100
max_steps = 100
parta_plt = np.zeros([n,total_episodes])

for i in range(0,n):
    # initialize current best reward, number of episodes and steps
    reward_best = -1  
    
    # initialize Q table to all zeros
    qtable = np.zeros([state_size, action_size])
     
    
    for episode in range(total_episodes):
        state = env.reset()
        step = 0
        done = False
        reward_tot = 0
        
        # take steps until an end result is reached
        for step in range(max_steps):
            
            # if there is no max in the current state, choose a random action
            if np.amax(qtable[state,:])==0:
                action = np.random.randint(0,4)
            else:   
                action = np.argmax(qtable[state,:])
                
            # take action and update qtable, reward and new state
            new_state, reward, done, info = env.step(action)
            qtable[state,action] = reward + 0.9*(np.amax(qtable[new_state,:]))
            reward_tot += reward       
            state = new_state
            if done == True: 
                break
    
        # update the best qtable if a better one is found
        if reward_tot > reward_best:
            reward_best = reward_tot
            qtable_best = qtable
            print(f'Part a, round {i}, better found - reward: {reward_best}')
            
        reward_avg = eval_policy(qtable_best,10,max_steps)
        parta_plt[i,episode] = reward_best
        
        # if episode == 99:
        #     print(qtable_best)
        

plt.figure()
plt.subplot(311)
plt.title('deterministic case, deterministic update rule')
# plt.xlabel('episodes')
# plt.ylabel('reward')
for i in range(0,n): 
    
    plt.plot(parta_plt[i,:], label=(f'episode {i}'))
# plt.show()



# -------non-deterministic case with deterministic update rule

env = gym.make("FrozenLake-v1", is_slippery=True)
env.reset()


# select number of episodes and max steps
# max steps are set to make sure program doesn't get stuck in a loop
total_episodes = 100
max_steps = 100
partb_plt = np.zeros([n,total_episodes])

for i in range(0,n):    
    # initialize reward
    reward_best = -1
    # initialize Q table to all zeros
    qtable = np.zeros([state_size, action_size])
    
    for episode in range(total_episodes):
        state = env.reset()
        step = 0
        done = False
        reward_tot = 0
        
        # take steps until an end result is reached
        for step in range(max_steps):
            
            # if there is no max in the current state, choose a random action
            if np.amax(qtable[state,:])==0:
                action = np.random.randint(0,4)
            else:   
                action = np.argmax(qtable[state,:])
                
            # take action and update qtable, reward and new state
            new_state, reward, done, info = env.step(action)
            qtable[state,action] = reward + 0.9*(np.amax(qtable[new_state,:]))
            reward_tot += reward       
            state = new_state
            if done == True: 
                break
    
        # update the best qtable if a better one is found
        if reward_tot > reward_best:
            reward_best = reward_tot
            qtable_best = qtable
            print(f'Part b, round {i}, better found - reward: {reward_best}')
            

        reward_avg = eval_policy(qtable_best,10,max_steps)
        partb_plt[i,episode] = reward_avg
        
        # if episode == 99:
        #     print(qtable_best)
      

plt.subplot(312)
plt.title('non-deterministic case, deterministic update rule')
# plt.xlabel('episodes')
# plt.ylabel('reward')
for i in range(0,n):  
    plt.plot(partb_plt[i,:], label=(f'episode {i}'))



# -------non-deterministic case with non-deterministic update rule

env.reset()

# select number of episodes and max steps
# max steps are set to make sure program doesn't get stuck in a loop
reward_best = -1
total_episodes = 100
max_steps = 100

partc_plt = np.zeros([n,total_episodes])

for i in range(0,n):
    # initialize reward
    reward_best = -1
    # initialize Q table to all zeros
    qtable = np.zeros([state_size, action_size])
    
    for episode in range(total_episodes):
        state = env.reset()
        step = 0
        done = False
        reward_tot = 0
        
        # take steps until an end result is reached
        for step in range(max_steps):
            
            # if there is no max in the current state, choose a random action
            if np.amax(qtable[state,:])==0:
                action = np.random.randint(0,4)
            else:   
                action = np.argmax(qtable[state,:])
                
            # take action and update qtable, reward and new state
            new_state, reward, done, info = env.step(action)
            qtable[state,action] = qtable[state,action] + 0.5*(reward + 
                                    0.9*(np.amax(qtable[new_state,:])) - 
                                    qtable[state, action])
            reward_tot += reward       
            state = new_state
            if done == True: 
                break
    
        # update the best qtable if a better one is found
        if reward_tot > reward_best:
            reward_best = reward_tot
            qtable_best = qtable
            print(f'Part c, round {i}, better found - reward: {reward_best}')
            

        reward_avg = eval_policy(qtable_best,10,max_steps)
        partc_plt[i,episode] = reward_avg
        
        # if episode == 99:
        #     print(qtable_best)
        

plt.subplot(313)
plt.title('non-deterministic case, non-deterministic update rule')
plt.xlabel('episodes')
plt.ylabel('reward')
for i in range(0,n):  
    plt.plot(partc_plt[i,:], label=(f'episode {i}'))

plt.tight_layout(pad=1.0)
plt.show()

