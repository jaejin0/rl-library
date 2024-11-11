'''
DDPG : Deep Deterministic Policy Gradient
Actor-Critic architecture

actor :
    input : observation
    output : action
    goal : maximize Q-value
critic : 
    input : observation, action
    output : Q-value
    goal : get accurate Q-value (reward + discount * next Q-value)
'''
