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

        self.lamda = 0.5

        self.qValue = util.Counter()
        self.eTrace = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        stateAction = (state, action)
        return self.qValue[stateAction]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        bestAction = self.computeActionFromQValues(state)
        if bestAction is not None:
            stateAction = (state, bestAction)
            return self.qValue[stateAction]
        else:
            return 0.0


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        # New 'local' qValue counter: action -> qvalue
        qValues = util.Counter()

        # If there are no legal action we return None
        possibleActions = self.getLegalActions(state)
        if possibleActions is None:
            return None

        # We save our qValues for our possible actions in the local qValue counter
        for action in possibleActions:
            qValues[action] = self.getQValue(state, action)

        # We pick the best actions, if multiple action have the same qValue then we pick random
        bestActions = []
        highestValue = qValues[qValues.argMax()]

        # We save all the actions with the highestValue in bestActions
        for action in qValues.keys():
            if (qValues[action] == highestValue):
                bestActions.append(action)

        # We return one action, either the only one or a random
        bestAction = None

        if (len(bestActions) > 1):
            bestAction = random.choice(bestActions)
        if (len(bestActions) == 1):
            bestAction = bestActions[0]

        return bestAction



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

        stateAction = (state, action)
        qvalue_current = self.qValue[stateAction]

        # Q-Learning - we take always take the action with the best q-value
        qvalue_next_qlearning = self.computeValueFromQValues(nextState)

        # SARSA - On-policy learning, we consider our policy (e-greedy) and we take that action and qvalue instead
        stateAction = (nextState, self.getAction(nextState))
        qvalue_next_sarsa = self.qValue[stateAction]

        delta = reward + (self.gamma * qvalue_next_sarsa) - qvalue_current

        self.eTrace[(state, action)] = self.eTrace[(state, action)] + 1

        for (state,action) in self.qValue.keys():
            self.qValue[(state, action)] = self.qValue[(state, action)] + self.alpha * delta * self.eTrace[(state,action)]
            self.eTrace[(state, action)] = self.gamma * self.lamda * self.eTrace[(state, action)]

        if (nextState == "TERMINAL STATE"):
            self.eTrace.clear()


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


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        if(reward != -1):
            self.eTrace.clear()

        qvalue_current = self.getQValue(state, action)
        qvalue_next = self.computeValueFromQValues(nextState)

        delta = reward + (self.gamma * qvalue_next) - qvalue_current

        features = self.featExtractor.getFeatures(state, action)
        for key in features.keys():
            self.eTrace[key] = self.lamda * self.eTrace[key] + features[key]
            self.weights[key] = self.weights[key] + self.alpha * delta * self.eTrace[key]



    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
