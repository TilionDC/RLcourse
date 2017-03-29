# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import numpy as np

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, **args):
        "You can initialize Q-values here..."
        "*** YOUR CODE HERE ***"
        ReinforcementAgent.__init__(self, **args)
        self.epsilon = epsilon

        # Discout factor
        self.gamma = gamma

        # Learning rate
        self.alpha = alpha
        self.qvalue = {}

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        qvalue = self.qvalue.get((state, action), None)
        # We haven't seen this state before
        if (qvalue == None):
            qvalue = 0.0
        return qvalue


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        best_qvalue = -99990
        # Best action(s) list
        for pair in self.qvalue:
            if (pair[0] == state):
                action = pair[1]
                if(self.getQValue(state, action) > best_qvalue):
                    best_qvalue = self.getQValue(state, action)
        if (best_qvalue == -99990):
            best_qvalue = 0.0
        return best_qvalue


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        best_qvalue = -999999
        best_actions = []
        best_action = None

        if not legalActions:
            best_action = None

        else:
            for action in legalActions:
                if (self.getQValue(state, action) > best_qvalue):
                    best_qvalue = self.getQValue(state, action)
                    best_actions = []
                    best_actions.append(action)
                if (self.getQValue(state, action) == best_qvalue):
                    best_actions.append(action)

        if (len(best_actions) > 1):
            best_action = random.choice(best_actions)
        if (len(best_actions) == 1):
            best_action = best_actions[0]

        return best_action



    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        "*** YOUR CODE HERE ***"
        if not legalActions:
            action = None
        else:
            if not util.flipCoin(self.epsilon):
                action = self.computeActionFromQValues(state)
            else:
                action = random.choice(legalActions)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        qvalue_current = self.qvalue.get((state, action), None)
        # print ("qvalue_current_before = " + str(qvalue_current))
        # We haven't seen this state before
        if (qvalue_current == None):
            self.qvalue[(state, action)] = 0.0
            qvalue_current = 0.0
        # print ("qvalue_current = " + str(self.qvalue[(state, action)]))

        qvalue_next = self.computeValueFromQValues(nextState)
        # print ("qvalue_next = " + str(qvalue_next))

        delta = reward + (self.gamma * qvalue_next) - qvalue_current
        self.qvalue[(state, action)] = qvalue_current + self.alpha * delta
        # print ("update = " + str(self.qvalue[(state, action)]))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        qvalue = np.dot(self.weights, features)
        return qvalue

    def computeValueFromQValues(self, state):
        legalAction = self.getLegalActions(state)
        best_qvalue = -999999
        for action in legalAction:
            qvalue = self.getQValue(state, action)
            if (qvalue > best_qvalue):
                best_qvalue = qvalue
        if (best_qvalue == -999999):
            best_qvalue = 0.0
        return best_qvalue


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        delta = reward + (self.gamma * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        for key in features.keys():
            value = features[key]
            if(key == "#-of-ghosts-1-step-away"):
                if (value != 0.0):
                    print ("feature \t" + str(key) + "\t" + str(value))
                    print ("current weights \t " + str(self.weights[key]))
                    print ("updated weights \t " + str(self.weights[key] + self.alpha * delta * value) + "\n")
            self.weights[key] = self.weights[key] + self.alpha * delta * value


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
