'''
Model of the evolution of identity covert signaling

Author: Matthew A. Turner
Date: 2020-01-23
'''
import numpy as np

from numpy.random import choice, uniform


RECEIVING_STRATEGIES = ["Generous", "Churlish"]
SENDING_STRATEGIES = ["Overt", "Covert"]


class Model:

    def __init__(self, N=100, n_rounds=10, K=3, prop_overt=0.5,
                 prop_covert=0.5, similarity_benefit=0.5,
                 one_dislike_penalty=0.25, two_dislike_penalty=0.25,
                 homophily=0.25):
        '''
        Arguments:
            N (int): Number of agents, i.e. population
            n_r (int): Number of assortment/payoff rounds per interaction round
            K (int): Number of traits per agent
            prop_overt (float): Proportion of agents receiving overt signals
            prop_covert (float): Proportion of agents receiving covert signals
            similarity_benefit (float): Benefit to dyads where there is some
                minimal similarity between agents. Should be between 0 and 1.
            one_dislike_penalty (float): Deduction in interaction payoff when
                one agent dislikes the other
            two_dislike_penalty (float): Further deduction added to the one above
                from a payoff of 1. The sum of one_dislike_penalty and
                two_dislike_penalty should be less than 1.0.
            homophily (float): the degree to which agents prefer to interact
                with similar others. Should be between 0 and 0.5.
        '''

        self.N = N
        self.n_rounds = n_rounds
        self.K = K
        self.prop_overt = prop_overt
        self.prop_covert = prop_covert
        self.similarity_benefit = similarity_benefit
        self.one_dislike_penalty = one_dislike_penalty
        self.two_dislike_penalty = two_dislike_penalty
        self.homophily = homophily

        assert (homphily >= 0.0) and (homophily <= 0.5)
        self.prob_likes_interact = 0.5 + homophily
        self.prob_dislikes_interact = 0.5 - homophily

        self.agents = [Agent() for _ in range(N)]

    def run(self, n_iter):
        '''
        Run n_iter rounds of interactions. One interaction is defined as
        n_r rounds of pairing and possible dyadic interaction between paired
        agents. After n_r rounds of pairing and dyadic interaction, evolution
        occurs where each agent potentially learns from a randomly chosen
        partner, with probability of learning defined in the same way as
        probability of interacting within a pairing of agents.
        '''

        for iter_idx in range(n_iter):
            for round_idx in range(self.n_rounds):
                self._run_round()

            self._evolve()

    def _run_round(self):

        self._signal_and_recieve()

        self._dyadic_interactions()

    def _signal_and_receive(self):

        for agent in self.agents:

            # Determine which of the two receiver proportions should be used.
            receive_prob = (
                self.prop_overt
                if agent.signaling_strategy == "Overt" else
                self.prop_covert
            )

            # Build a list of receivers who will observe the signal.
            receivers = [
                other for other in self.agents
                if (other != agent) and (uniform() < receive_prob)
            ]

            # TODO: implement signaling by focal `agent` and receiving by
            # other agents.

    def _dyadic_interactions(self):

        # Make potential interaction dyads.
        pairs = choice(self.agents, size=(self.N/2, 2), replace=False)

        # Probabilistic dyadic interaction.
        interacting_pairs = [
            pair for pair in pairs
            if uniform() < self._dyadic_interaction_prob(*pair)
        ]

        # Calculate payoffs for each pair who interact and add to each
        # agent's cumulative payoff.
        for pair in interacting_pairs:
            payoff = _caluclate_payoff(*pair)
            for a in pair:
                a.payoff += payoff

    def _dyadic_interaction_prob(self, agent1_idx, agent2_idx):
        a1 = self.agents[agent1_idx]
        a2 = self.agents[agent2_idx]

        a1_att = a1.attitudes[agent2_idx]
        a2_att = a2.attitudes[agent1_idx]

        if a1_att == 1:
            if a2_att == 1:
                return self.prob_likes_interact
            elif a2_att == 0:
                return (self.prob_likes_interact + 0.5) / 2.0
            elif a2_att == -1:
                return 0.5
            else:
                raise RuntimeError("Attitudes must be 1, 0, or -1")

        elif a1_att == 0:
            if a2_att == 1:
                return (self.prob_likes_interact + 0.5) / 2.0
            elif a2_att == 0:
                return 0.5
            elif a2_att == -1:
                return (self.prob_dislikes_interact + 0.5) / 2.0
            else:
                raise RuntimeError("Attitudes must be 1, 0, or -1")

        elif a1_att == -1:
            if a2_att == 1:
                return 0.5
            elif a2_att == 0:
                return (self.prob_dislikes_interact + 0.5) / 2.0
            elif a2_att == -1:
                return self.prob_dislikes_interact
            else:
                raise RuntimeError("Attitudes must be 1, 0, or -1")

        else:
            raise RuntimeError("Attitudes must be 1, 0, or -1")

    def _evolve(self):
        pass


class Agent:

    def __init__(self, K=3, N=100):
        '''
        Agent initialization is fully random in this model.
        '''

        # Agents initially have K binary traits. ±1 is used for determining
        # similarity or dissimilarity with other agents via summation.
        self.traits = choice([-1, 1], size=K)

        # Agents are either Generous or Churlish, and this determines their
        # initial attitudes towards other agents.
        self.receiving_strategy = choice(RECEIVING_STRATEGIES)

        # Generous agents are neutral towards unknown others.
        if self.receiving_strategy == "Generous":
            attitudes = np.zeros((N,), dtype=int)
        # Churlish agents dislike unknown others.
        else:
            attitudes = -1 * np.ones((N,), dtype=int)

        # Set agent signaling strategy.
        self.signaling_strategy = choice(SENDING_STRATEGIES)

        # Payoffs initially 0, but will accumulate over time.
        self.payoff = 0.0
