
# coding: utf-8

from itertools import product
import random


import numpy as np
from matplotlib import pyplot as plt


from COVIS_R_based_explicit import RuleBasedSystem;
from COVIS_procedural_system import ProceduralLearningSystem;


class COVIS(object):
    """
    COVIS has two subsystems: (E)xplicit and (P)rocedural. At each stimulus presentation, COVIS 
    feeds the stimulus to both subsystems and then decides whose prediction to use based on two 
    parameters:
    1. Theta - trust in a given subsystem based on its success history
    2. h - confidence of a given sybsystem in its prediction
    """
    
    def __init__(self, delta_OC, delta_OE, rule_based_params, procedural_params):
        self.delta_OC = delta_OC
        self.delta_OE = delta_OE
        self.rule_based_system = RuleBasedSystem(**rule_based_params)
        self.procedural_system = ProceduralLearningSystem(**procedural_params)
        
        # Initial values for trust (p. 77)
        self.Theta_E = 0.99
        self.Theta_P = 0.01
        
        self.current_prediction_E = None
        self.current_prediction_P = None
        self.current_winner = None
        
    def _feed_stimulus_to_a_system(self, stimulus, system, real_categ):
        """
        Returns (confidence, prediction) of system
        """
        system.process_stimulus(stimulus, real_categ)
        return 0.5, system.current_prediction
        
    def process_stimulus(self, stimulus, real_categ):
        
        # h_* - absolute value of the normalized discriminant value, not the original value (see p. 76)
        h_E, self.current_prediction_E = self._feed_stimulus_to_a_system(
            stimulus, self.rule_based_system, real_categ)
        h_P, self.current_prediction_P = self._feed_stimulus_to_a_system(
            stimulus, self.procedural_system, real_categ)
        
        # Select whose prediction to use (p. 77)
        if h_E * self.Theta_E > h_P * self.Theta_P:
            self.current_prediction = self.current_prediction_E
            self.predicted_by = self.rule_based_system
        else:
            self.current_prediction = self.current_prediction_P
            self.predicted_by = self.rule_based_system
            
        
        self.process_feedback(real_categ)
        is_correct = self.current_prediction == real_categ
        return is_correct
            
    def process_feedback(self, feedback):
        """
        Updates the trust values
        """
        if self.current_prediction_E == feedback:
            self.Theta_E += self.delta_OC * (1 - self.Theta_E)  # p. 77, Eq. (15)
        else:
            self.Theta_E -= self.delta_OE * self.Theta_E  # p. 77, Eq. (16)
            
        self.Theta_P = 1 - self.Theta_E  # p. 77, paragraph after Eq. (16)        
        
    


# ![image.png](attachment:image.png)

rule_based_params = dict(
    n_dims=4, sigma_e_2=0.0, delta_criterion=0,
    gamma=1.0, lambda_=5.0, 
    delta_C=0.0025, delta_E=0.02, 
)

stimuli = list(product((0, 1), repeat=4))
procedural_params = dict(
    inp_preferred_stimuli=np.array(stimuli), 
    input_scale=0.01,
    categs=("A", "B"), sigma_striatal=0.0125,
    theta_nmda=0.0022, theta_ampa=0.01, d_base=0.2,
    alpha=0.65, beta=0.19, gamma=0.02,
    w_max=1.0
)


covis = COVIS(
    delta_OC=0.01, 
    delta_OE=0.04,
    rule_based_params=rule_based_params,
    procedural_params=procedural_params
             )


stimuli = list(product((0, 1), repeat=4))
n_stim = len(stimuli)

rb_categs = ["B" if stim[0] == 1 else "A" for stim in stimuli]

n_blocks = 10
blocks_rb = []
for _iter in range(n_blocks):
    block_hits = []
    for ind in random.sample(range(n_stim), n_stim):
        block_hits.append(covis.process_stimulus(stimuli[ind], rb_categs[ind]))
    blocks_rb.append(sum(block_hits) / float(n_stim))


plt.plot(blocks_rb, 'b', label='rule-based')

plt.xlabel('Blocks')
plt.ylabel('Accuracy')

plt.axis([0, n_blocks - 1, 0, 1.1])
plt.legend()

plt.show()

