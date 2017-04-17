# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        
        AllStates  = mdp.getStates()
        self.values = util.Counter()

        #Iterate value function
        for k in range(iterations):
            Vnew = util.Counter() #Batch computation
            for State in AllStates:
                
                AllActions = mdp.getPossibleActions(State)
                if len(AllActions) > 0: #Some actions are available
                    ExpectedValueofAction = util.Counter() #Temporary counter of value of each action available at s
                    
                   
                    for Action in AllActions:
                        
                        Pssa = mdp.getTransitionStatesAndProbs(State,Action) #List of ((s'), probability) for s,a
                        for Transition in Pssa: #sum over all possible s' = StatePrime
                            StatePrime  = Transition[0]
                            Probability = Transition[1]
                            Reward      = mdp.getReward(State,Action,StatePrime)
                            Vprime      = self.values[StatePrime]
                            ExpectedValueofAction[Action] += Probability*(Reward + discount*Vprime)
                    
                    #Pick the best action in ValueofActions:
                    SortedActions = ExpectedValueofAction.sortedKeys()
                    OptimalAction = SortedActions[0]
                    
                    #print "State :"+str(State)+" | Optimal Action: "+OptimalAction
            
                    #Update value function
                    Vnew[State] = ExpectedValueofAction[OptimalAction]
            
                #else: no available action -> don't do anything to self.values[State]
                
            self.values = Vnew
                    
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        
        Pssa = self.mdp.getTransitionStatesAndProbs(state,action)
        
        ExpectedValueofAction = 0
        for Transition in Pssa: #sum over all possible s' = StatePrime
            StatePrime  = Transition[0]
            Probability = Transition[1]
            Reward      = self.mdp.getReward(state,action,StatePrime)
            Vprime      = self.values[StatePrime]
            ExpectedValueofAction += Probability*(Reward + self.discount*Vprime)
        
        return ExpectedValueofAction
            
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        
        "*** YOUR CODE HERE ***"
        AllActions = self.mdp.getPossibleActions(state)
        if len(AllActions) > 0: #Some actions are available
            ExpectedValueofAction = util.Counter() #Temporary counter of value of each action available at s
            for Action in AllActions:
                
                ExpectedValueofAction[Action] = self.computeQValueFromValues(state, Action)
            
            #Pick the best action in ValueofActions:
            OptimalAction = ExpectedValueofAction.argMax()
        else:
            OptimalAction =  'None'
        

        return OptimalAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
