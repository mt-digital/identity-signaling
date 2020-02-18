'''
Model of the evolution of identity covert signaling

Author: Matthew A. Turner
Date: 2020-01-23
'''
import numpy as np

from numpy.random import choice, uniform
from scipy.special import expit


RECEIVING_STRATEGIES = ["Generous", "Churlish"]
SIGNALING_STRATEGIES = ["Overt", "Covert"]


class Model:

    def __init__(self, N=100, n_rounds=10, K=3, prob_overt_receiving=0.75,
                 prob_covert_receiving=0.25, similarity_benefit=0.25,
                 one_dislike_penalty=0.25, two_dislike_penalty=0.25,
                 homophily=0.25, evo_logistic_loc=1.25, random_seed=None,
                 evo_logistic_scale=12):
        '''
        Arguments:
            N (int): Number of agents, i.e. population
            n_r (int): Number of assortment/payoff rounds per interaction round
            K (int): Number of traits per agent
            prob_overt_receiving (float): Proportion of agents receiving overt signals
            prob_covert_receiving (float): Proportion of agents receiving covert signals
            similarity_benefit (float): Benefit to dyads where there is some
                minimal similarity between agents. Should be between 0 and 1.
            one_dislike_penalty (float): Deduction in interaction payoff when
                one agent dislikes the other
            two_dislike_penalty (float): Further deduction added to the one above
                from a payoff of 1. The sum of one_dislike_penalty and
                two_dislike_penalty should be less than 1.0.
            homophily (float): the degree to which agents prefer to interact
                with similar others. Should be between 0 and 0.5.
            evo_logistic_loc (float): location where logistic function = 0.5
                probability of switching strategies depending on relative
                payoff. I.e. default is set so that 50% chance of switching
                when the teacher has accumulated 1.25x more than the learner.
            evo_logistic_scale (float): scale defining the sharpness of the
                transition from 0 to 1 in the logistic function. Default chosen
                so that probability of switching is ~1/50 when
                relative payoff is 0.9 and ~3/50 when relative payoff is 1.0.
        '''

        self.N = N
        self.n_rounds = n_rounds
        self.K = K
        self.prob_overt_receiving = prob_overt_receiving
        self.prob_covert_receiving = prob_covert_receiving
        self.similarity_benefit = similarity_benefit
        self.one_dislike_penalty = one_dislike_penalty
        self.two_dislike_penalty = two_dislike_penalty
        self.homophily = homophily
        self.evo_logistic_loc = evo_logistic_loc
        self.evo_logistic_scale = evo_logistic_scale

        assert (homophily >= 0.0) and (homophily <= 0.5)

        np.random.seed(random_seed)

        self.agents = [Agent(idx, K=K, N=N) for idx in range(N)]

        # This is the series of proportion of covert signalers.
        self.prop_covert_series = np.array([_proportion_covert(self)])
        self.prop_churlish_series = np.array([_proportion_churlish(self)])


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

            self._signal_and_receive()

            for round_idx in range(self.n_rounds):
                self._dyadic_interactions()

            self._evolve()

            self.prop_covert_series= np.append(
                self.prop_covert_series, _proportion_covert(self)
            )
            self.prop_churlish_series= np.append(
                self.prop_churlish_series, _proportion_churlish(self)
            )

            self._reset_attitudes()

    def _signal_and_receive(self):

        for signaller_idx, signaller in enumerate(self.agents):

            # Determine which of the two receiver proportions should be used.
            receive_prob = (
                self.prob_overt_receiving
                if signaller.signaling_strategy == "Overt" else
                self.prob_covert_receiving
            )

            # Build a list of receivers who will observe the signal.
            receivers = (
                other for other in self.agents
                if (other != signaller) and (uniform() < receive_prob)
            )

            # TODO: implement signaling by focal `agent` and receiving by
            # other agents.
            for receiver in receivers:

                # Calculate similarity and set whether receiver likes
                # signaling agent. "Liking" is determined by how many traits
                # they have in common---if they have a majority of their traits
                # in common they like the signaling agent. Neutral if K even
                # and equal number of similar and dissimilar traits.
                # XXX This can be calculated just once at init after
                # traits have been set since traits don't change.
                similarity = np.sum(receiver.traits * signaller.traits)

                # If similarity is positive, attitude independent of signaling
                # strategy...
                if similarity > 0:
                    receiver.attitudes[signaller_idx] = 1
                # ...but if similarity is negative, need to check the
                # signaling strategy.
                elif similarity < 0:
                    # If the signal was overt, change attitude to dislike.
                    # In the other case, do nothing as this will keep the
                    # default attitude towards the signaler by the receiving
                    # agent.
                    if signaller.signaling_strategy == "Overt":
                        receiver.attitudes[signaller_idx] = -1

    def _dyadic_interactions(self):

        # Make potential interaction dyads.
        pairs = choice(self.agents, size=(self.N//2, 2), replace=False)

        # Probabilistic dyadic interaction.
        interacting_pairs = [
            pair for pair in pairs
            if uniform() < self._dyadic_interaction_prob(*pair)
        ]

        # Calculate payoffs for each pair who interact and add to each
        # agent's cumulative payoff. Increase count of number of interactions
        # for both agents in pair.
        for pair in interacting_pairs:
            payoff = self._calculate_payoff(*pair)
            for a in pair:
                a.gross_payoff += payoff
                a.n_interactions += 1

            # Need to add partner to list of previous partners
            p0 = pair[0]
            p1 = pair[1]

            p0_partners = pair[0].previous_partners
            p1_partners = pair[1].previous_partners

            if p0 not in p1_partners:
                p1_partners.add(p0.index)

            if p1 not in p0_partners:
                p0_partners.add(p1.index)

    def _dyadic_interaction_prob(self, a1, a2):

        a1_att = a1.attitudes[a2.index]
        a2_att = a2.attitudes[a1.index]

        att_sum = a1_att + a2_att

        return 0.5 + (self.homophily * att_sum / 2.0)

    def _evolve(self):
        # TODO: implement learning selection using _dyadic_interaction_prob
        # and according to process outlined in model document.
        for learner in self.agents:
            # Select teacher at random and re-select if teacher is focal agent.
            # XXX this is not quite what the model spec says. Partner
            # selection should also involve homophily using
            # _dyadic_interaction_prob somehow. Maybe each agent's choice
            # should be weighted by the dyadic interaction prob.

            # Calculate interaction probability for every possible teacher,
            # setting self-teaching probability to zero.
            probs = np.array(
                [
                    self._dyadic_interaction_prob(learner, maybe_teacher)
                    if maybe_teacher != learner else 0.0

                    for maybe_teacher in self.agents
                ]
            )
            # Normalize probabilities.
            probs = probs / probs.sum()
            # Weight random teacher selection by calculated probabilities.
            teacher = choice(self.agents, p=probs)

            # Learner payoff sometimes is zero at the beginning of the model...
            if learner.payoff > 0:
                payoff_proportion = teacher.payoff / learner.payoff
            # ... if it is, set chance to switch to be 0.5.
            else:
                payoff_proportion = self.evo_logistic_loc

            switch_prob = _logistic(payoff_proportion,
                                    loc=self.evo_logistic_loc,
                                    scale=self.evo_logistic_scale)

            if uniform() < switch_prob:
                # Coin flip to switch either signaling or receiving strategy.
                strategy_type = (
                    "Signaling" if 0.5 < uniform() else "Receiving"
                )
                # Switch specified teacher strategy for learner's in-place.
                # XXX setting strategy type explicitly for now.
                # strategy_type = "Signaling"
                # print(strategy_type)
                if strategy_type == "Signaling":
                    learner.signaling_strategy = teacher.signaling_strategy
                else:
                    learner.receiving_strategy = teacher.receiving_strategy

    def _reset_attitudes(self):
        '''
        This resets attitudes, which occurs after evolution happens.
        '''
        for agent in self.agents:
            if agent.receiving_strategy == "Generous":
                self.attitudes = np.zeros((self.N,), dtype=int)
            else:
                self.attitudes = -1 * np.ones((self.N,), dtype=int)


    def _calculate_payoff(self, a1, a2):
        '''
        Based on agent attitudes and similarity in traits, calcuate the
        payoff resulting from an interaction between them.
        '''

        att_sum = a1.attitudes[a2.index] + a2.attitudes[a1.index]

        # Like/like.
        if att_sum == 2:
            return 1 + self.similarity_benefit

        # Like/neutral.
        elif att_sum == 1:
            return 1 + self.similarity_benefit

        # Neutral/neutral and like/dislike.
        elif att_sum == 0:
            # Neutral/neutral.
            if a1.attitudes[a2.index] == 0 and a2.attitudes[a1.index] == 0:
                return 1 + self.similarity_benefit
            # Like/dislike.
            elif a1.attitudes[a2.index] == 1 or a2.attitudes[a1.index] == 1:
                return 1 + self.similarity_benefit - self.one_dislike_penalty
            else:
                raise RuntimeError("An unexpected attitude situation occurred")

        # Neutral/dislike.
        elif att_sum == -1:
            return 1 - self.one_dislike_penalty

        # Dislike/dislike.
        elif att_sum == -2:
            return 1 - self.one_dislike_penalty - self.two_dislike_penalty

        # Shouldn't get here, but...
        else:
            raise RuntimeError(
                "The attitude sum was not bounded between -2 and 2"
            )


def _logistic(x, loc=0, scale=1):
    '''
    Private method to make a logistic function with loc and scale out of the
    equivalent scipy special function `expit`, which lacks them:
    https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.special.expit.html

    Arguments:
        x (float or np.ndarray): input to logistic function
        loc (float): x value where logistic function is 0.5
        scale (float): multiplicative coefficient on x that sets the sharpness
            of the transition between 0 and 1

    Returns:
        (float or np.ndarray): output of the logistic function
    '''
    xtrans = scale * (x - loc)
    return expit(xtrans)

#: Calculate proportion of covert agents in a Model instance.
def _proportion_covert(model):
    return (
        np.sum(
            [a.signaling_strategy == "Covert" for a in model.agents]
        ) / model.N
    )


#: Calculate proportion of churlish agents in a Model instance.
def _proportion_churlish(model):
    return (
        np.sum(
            [a.receiving_strategy == "Churlish" for a in model.agents]
        ) / model.N
    )

class Agent:

    def __init__(self, agent_idx=0, K=3, N=100,
                 receiving_strategy=None, signaling_strategy=None):
        '''
        Agent initialization is fully random in this model.
        '''
        self.index = agent_idx

        # Agents initially have K binary traits. ±1 is used for determining
        # similarity or dissimilarity with other agents via summation.
        self.traits = choice([-1, 1], size=K)

        # Agents are either Generous or Churlish, and this determines their
        # initial attitudes towards other agents.
        if receiving_strategy is None:
            self.receiving_strategy = choice(RECEIVING_STRATEGIES)
        else:
            assert receiving_strategy in RECEIVING_STRATEGIES
            self.receiving_strategy = receiving_strategy

        # Generous agents are neutral towards unknown others.
        if self.receiving_strategy == "Generous":
            self.attitudes = np.zeros((N,), dtype=int)
        # Churlish agents dislike unknown others.
        else:
            self.attitudes = -1 * np.ones((N,), dtype=int)

        # Set agent signaling strategy.
        if signaling_strategy is None:
            self.signaling_strategy = choice(SIGNALING_STRATEGIES)
        else:
            assert signaling_strategy in SIGNALING_STRATEGIES
            self.signaling_strategy = signaling_strategy

        # Payoffs initially 0, but will accumulate over time.
        self.gross_payoff = 0.0

        # Total number of interactions agent has had.
        self.n_interactions = 0

        # Remember who I have interacted with by their index.
        self.previous_partners = set()

    @property
    def payoff(self):
        # Avoid divide by zero. If agent has not interacted, no payoffs yet.
        if self.n_interactions > 0:
            return self.gross_payoff / self.n_interactions
        else:
            return 0
