'''
Model of the evolution of identity covert signaling

Author: Matthew A. Turner
Date: 2020-02-25
'''
import numpy as np
import warnings

from copy import deepcopy
from numpy.random import choice, uniform, shuffle
from scipy.special import expit
from scipy.spatial.distance import hamming


RECEIVING_STRATEGIES = ['Generous', 'Churlish']
SIGNALING_STRATEGIES = ['Overt', 'Covert']


class Model:

    def __init__(self, N=100, n_rounds=100, K=9, prob_overt_receiving=1.0,
                 prob_covert_receiving=0.5, similarity_benefit=0.25,
                 one_dislike_penalty=0.25, two_dislike_penalty=0.25,
                 homophily=0.25, random_seed=None, similarity_threshold=0.5,
                 minority_trait_frac=None,
                 initial_prop_covert=0.5, initial_prop_churlish=0.5,
                 n_minmaj_traits=None,  # only used if minority_trait_frac ! None
                 learning_alpha=0.0, learning_beta=10.0):
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
            similarity_threshold (float): the minimum difference between the
                number of traits agents have in common and number of
                opposing traits
            minority_trait_frac (float): number between 0 and 1 indicating
                what fraction of agents should be ``minority'' agents with
                first trait +1, and set first trait of majority to -1. If
                None, do not set minority agents.
            n_minmaj_traits (int): number of traits to use in assigning
                minority and majority agents; short name is M in paper and
                CLI.
            learning_alpha (float): location where logistic function = 0.5
                probability of switching strategies depending on relative
                payoff. I.e. default is set so that 50% chance of switching
                when the teacher has accumulated 1.25x more than the learner.
            learning_beta (float): scale defining the sharpness of the
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
        self.similarity_threshold = similarity_threshold
        self.learning_alpha = learning_alpha
        self.learning_beta = learning_beta

        if n_minmaj_traits is None:
            n_minmaj_traits = K - ((K+1) // 2)
        self.n_minmaj_traits = n_minmaj_traits

        self.initial_prop_covert = initial_prop_covert
        self.initial_prop_churlish = initial_prop_churlish

        assert (homophily >= 0.0) and (homophily <= 0.5)

        if random_seed is not None:
            np.random.seed(random_seed)

        n_initial_churlish = int(N * initial_prop_churlish)
        n_initial_generous = N - n_initial_churlish
        n_initial_covert = int(N * initial_prop_covert)
        n_initial_overt = N - n_initial_covert

        # Create array of receiving strategies and randomize to vector-assign
        # to agents for initialization.
        rec_strategies = (
            ['Churlish'] * n_initial_churlish +
            ['Generous'] * n_initial_generous
        )
        np.random.shuffle(rec_strategies)

        # Create array of signaling strategies and randomize to vector-assign
        # to agents for initialization.
        sig_strategies = (
            ['Covert'] * n_initial_covert +
            ['Overt'] * n_initial_overt
        )
        np.random.shuffle(sig_strategies)

        self.agents = [

            Agent(idx, K=K, N=N,
                  receiving_strategy=rec_strategies[idx],
                  signaling_strategy=sig_strategies[idx])

            for idx in range(N)
        ]

        # Have a marker for the run() method if it should track majority/
        # minority agents over time so we can later recover their
        # proportional strategies.
        self.minority_test = False
        if minority_trait_frac is not None:

            # Set minority_trait_frac of agents to have first trait +1
            # and set the rest to have -1. The +1 will be the minorities
            # along the first trait dimension.
            self.minority_test = True

            # Select minority agents.
            self.minority_agents = list(choice(self.agents,
                                        size=int(N*minority_trait_frac),
                                        replace=False))

            # Majority agents are the ones unselected for minority.
            self.majority_agents = list(
                set(self.agents) - set(self.minority_agents)
            )

            # Set traits.
            for agent in self.minority_agents:
                agent.traits[0:n_minmaj_traits] = 1
            for agent in self.majority_agents:
                agent.traits[0:n_minmaj_traits] = -1

            self.prop_covert_series_minority = np.array(
                [_proportion_covert(self, subset='minority')]
            )
            self.prop_churlish_series_minority = np.array(
                [_proportion_churlish(self, subset='minority')]
            )

            self.prop_covert_series_majority = np.array(
                [_proportion_covert(self, subset='majority')]
            )
            self.prop_churlish_series_majority = np.array(
                [_proportion_churlish(self, subset='majority')]
            )

        # This is the series of proportion of covert signalers.
        self.prop_covert_series = np.array([_proportion_covert(self)])
        self.prop_churlish_series = np.array([_proportion_churlish(self)])

        # Initialize similarity matrix of shape NxN to hold boolean
        # indicating whether or not two agents are similar.
        self.similar_matrix = np.zeros((N, N), dtype=bool)

        # It is repetetive but convenient to create the full symmetric matrix.
        self._init_similar_matrix()

        # These are only defined after communication round has concluded. We
        # store/calculate all N^2 instead of N(N-1)/2 one could do since
        # there is symmetry in interaction factors.
        self.interaction_factors = {
            (a1.index, a2.index): None
            for a1 in self.agents
            for a2 in self.agents
        }
        self._calculate_interaction_factors()

    def _init_similar_matrix(self):
        for a1 in self.agents:
            for a2 in self.agents:

                similarity = 1 - hamming(a1.traits, a2.traits)
                similar = similarity >= self.similarity_threshold

                self.similar_matrix[a1.index, a2.index] = similar

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

            for agent in self.agents:
                agent.gross_payoff = 0.0

            self._signal_and_receive()

            # Interaction factors determine probability one agent picks another
            # as an interaction partner. The probability cannot be determined
            # now, but the factors can. With N=100, this saves
            # 99 * 97 * ... * 3 operations for every interaction round. Add
            # another factor of 10 when there are ten interactions per
            # time step.
            self._calculate_interaction_factors()

            for round_idx in range(self.n_rounds):
                self._dyadic_interactions()

            self._social_learning()

            self.prop_covert_series = np.append(
                self.prop_covert_series, _proportion_covert(self)
            )
            self.prop_churlish_series = np.append(
                self.prop_churlish_series, _proportion_churlish(self)
            )

            # Record minority/majority agents if running the minority test.
            if self.minority_test:

                self.prop_covert_series_minority = np.append(
                    self.prop_covert_series_minority,
                    _proportion_covert(self, subset='minority')
                )
                self.prop_churlish_series_minority = np.append(
                    self.prop_churlish_series_minority,
                    _proportion_churlish(self, subset='minority')
                )

                self.prop_covert_series_majority = np.append(
                    self.prop_covert_series_majority,
                    _proportion_covert(self, subset='majority')
                )
                self.prop_churlish_series_majority = np.append(
                    self.prop_churlish_series_majority,
                    _proportion_churlish(self, subset='majority')
                )

            self._reset_attitudes()

    def _signal_and_receive(self):

        for signaler_idx, signaler in enumerate(self.agents):

            # Determine which of the two receiver proportions should be used.
            receive_prob = (
                self.prob_overt_receiving
                if signaler.signaling_strategy == "Overt" else
                self.prob_covert_receiving
            )

            # Build a list of receivers who will observe the signal.
            receivers = (
                other for other in self.agents
                if (other != signaler) and (uniform() < receive_prob)
            )

            # TODO: implement signaling by focal `agent` and receiving by
            # other agents.
            for receiver in receivers:

                # Calculate similarity and set whether receiver likes
                # signaling agent.
                similar = self._are_similar(signaler, receiver)

                # If similarity is positive, attitude independent of signaling
                # strategy...
                # if similarity > 0:
                if similar:
                    receiver.attitudes[signaler.index] = 1
                # ...but if similarity is negative, need to check the
                # signaling strategy.
                else:
                    # If the signal was overt, change attitude to dislike.
                    # In the other case, do nothing as this will keep the
                    # default attitude towards the signaler by the receiving
                    # agent.
                    if signaler.signaling_strategy == "Overt":
                        receiver.attitudes[signaler.index] = -1

    def _calculate_interaction_factors(self):

        # This calculates the factor for both permutations of agent pairings,
        # which is redundant, but simple to implement and understand and
        # we only have N~100, NxN = 10e4. If N=1e3 and NxN=1e6 maybe there
        # will be a problem, but I don't think 1mil elements is too much
        # of a problem if dicts pre-initialized.
        for a1 in self.agents:
            for a2 in self.agents:
                a1_att = a1.attitudes[a2.index]
                a2_att = a2.attitudes[a1.index]

                att_sum = a1_att + a2_att

                factor = 0.5 + (self.homophily * att_sum / 2.0)

                self.interaction_factors[(a1.index, a2.index)] = factor

    def _dyadic_interactions(self):

        # Make potential interaction dyads.
        # pairs = choice(self.agents, size=(self.N//2, 2), replace=False)
        dyads = self._make_dyads()

        # Calculate payoffs for each pair who interact and add to each
        # agent's cumulative payoff. Increase count of number of interactions
        # for both agents in pair.
        for dyad in dyads:
            payoff = self._calculate_payoff(*dyad)
            for a in dyad:
                a.gross_payoff += payoff
                a.n_interactions += 1

    def _dyadic_interaction_factor(self, a1, a2):

        return self.interaction_factors[a1.index, a2.index]
        # a1_att = a1.attitudes[a2.index]
        # a2_att = a2.attitudes[a1.index]

        # att_sum = a1_att + a2_att

        # return 0.5 + (self.homophily * att_sum / 2.0)

    def _interaction_probs(self, agent, available_others):
        '''
        Arguments:
            agent (Agent): Focal agent choosing an interaction partner.
            available_others ([Agent]): Other agents who have not yet in
                an interacting dyad.
        '''
        factors = np.zeros(self.N)
        for other in available_others:
            factors[other.index] = \
                self._dyadic_interaction_factor(agent, other)

        # Return either normalized factors or a randomly selected other having
        # a factor/probability 1 if
        # normalization denominator is zero for that focal agent,
        # indicating all factors are zero
        # in this case, which may happen if there is mutual dislike between
        # one agent and all others.
        denom = factors.sum()
        if denom > 0:
            return factors / factors.sum()
        elif denom == 0:
            other_indexes = [agent.index for agent in available_others]
            other_index = choice(other_indexes)
            factors[other_index] = 1.0
            return factors
        else:
            raise RuntimeError('Negative interaction probability factors!')

    def _make_dyads(self):

        # Need to draw agents for pairing from a copy of list of agents,
        # finishing once there are no more available agents.
        available_agent_indexes = [agent.index for agent in self.agents]

        # Initialize empty list of dyads to which we'll add in loop below.
        dyads = []

        # As long as there are more available agents, keep making dyads.
        # XXX Assumes even number of agents.
        while available_agent_indexes:
            # Randomly select focal agent to choose an interaction partner.
            focal_agent_index = choice(available_agent_indexes)
            focal_agent = self.agents[focal_agent_index]
            available_agent_indexes.remove(focal_agent_index)

            # Select interaction partner for focal agent.
            available_agents = \
                [self.agents[index] for index in available_agent_indexes]

            interaction_probs = \
                self._interaction_probs(focal_agent, available_agents)

            partner = choice(self.agents, p=interaction_probs)
            available_agent_indexes.remove(partner.index)

            dyads.append((focal_agent, partner))

        return dyads

        # XXX Placeholder; need more advanced algorithm to make one pair at
        # a time with .
        # return choice(self.agents, size=(self.N//2, 2), replace=False)

    def _social_learning(self):
        '''
        Returns: None
        '''
        # Learners are paired at random with teachers, then the learner
        # decides which strategy to adopt in maybe_update_strategy: the learner
        # either keeps its existing strategy or updates its strategy with the
        # teacher's strategy.
        for learner in self.agents:
            # TODO add interaction probabilities to test case where
            # w≠0. Here this implicitly assumes no homophily for teacher
            # selection.
            if self.minority_test and learner in self.minority_agents:
                teachers = [a for a in self.minority_agents
                            if a != learner]
            else:
                teachers = [a for a in self.agents
                            if a != learner]

            teacher = choice(teachers)

            # maybe_update_strategy will set the learner's next_strategy
            # attribute, used after all learners
            learner.maybe_update_strategy(
                teacher, learning_beta=self.learning_beta
            )

        # Go through all agents and update strategies according to what is
        # contained in the Agent's `next_strategy` attribute.
        for agent in self.agents:
            # next_strategy is a dict with key that is the strategy type,
            # signaling or receiving.
            if agent.next_strategy is not None:
                strategy_type = list(agent.next_strategy.keys())[0]
                if strategy_type == 'signaling':
                    agent.signaling_strategy = \
                        agent.next_strategy[strategy_type]
                if strategy_type == 'receiving':
                    agent.receiving_strategy = \
                        agent.next_strategy[strategy_type]

    def _reset_attitudes(self):
        '''
        This resets attitudes, which occurs after evolution happens.
        '''
        for agent in self.agents:
            if agent.receiving_strategy == "Generous":
                self.attitudes = np.zeros((self.N,), dtype=int)
            else:
                self.attitudes = -1 * np.ones((self.N,), dtype=int)

    def _are_similar(self, a1, a2):
        return self.similar_matrix[a1.index, a2.index]

    def _calculate_payoff(self, a1, a2):
        '''
        Based on agent attitudes and similarity in traits, calcuate the
        payoff resulting from an interaction between them.
        '''

        att_sum = a1.attitudes[a2.index] + a2.attitudes[a1.index]

        # XXX OLD
        # similar = (np.sum(a1.traits * a2.traits) >= self.similarity_threshold)
        similar = self._are_similar(a1, a2)

        # Like/like.
        if att_sum == 2:
            return 1 + self.similarity_benefit

        # Like/neutral.
        elif att_sum == 1:
            return 1 + self.similarity_benefit

        # Neutral/neutral and like/dislike.
        elif att_sum == 0:
            if similar:
                # Like/dislike
                if a1.attitudes[a2.index] > 0 or a2.attitudes[a1.index] > 0:
                    return 1 + self.similarity_benefit - self.one_dislike_penalty
                # Neutrals
                elif a1.attitudes[a2.index] == 0 and a2.attitudes[a1.index] == 0:
                    return 1 + self.similarity_benefit
            else:
                # Like/dislike XXX IS THIS EVEN POSSIBLE? smelly....
                if a1.attitudes[a2.index] > 0 or a2.attitudes[a1.index] > 0:
                    return 1 - self.one_dislike_penalty

                # Neutrals
                elif a1.attitudes[a2.index] == 0 and a2.attitudes[a1.index] == 0:
                    return 1

        # Neutral/dislike.
        elif att_sum == -1:
            if similar:
                return 1 + self.similarity_benefit - self.one_dislike_penalty
            else:
                return 1 - self.one_dislike_penalty

        # Dislike/dislike.
        elif att_sum == -2:
            if similar:
                return 1 + self.similarity_benefit \
                         - self.one_dislike_penalty - self.two_dislike_penalty
            else:
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
def _proportion_covert(model, subset=None):

    if subset is None:
        return (
            np.sum(
                [a.signaling_strategy == "Covert" for a in model.agents]
            ) / model.N
        )
    elif subset == 'minority':
        return (
            np.sum(
                [a.signaling_strategy == "Covert"
                 for a in model.minority_agents]
            ) / len(model.minority_agents)
        )
    elif subset == 'majority':
        return (
            np.sum(
                [a.signaling_strategy == "Covert"
                 for a in model.majority_agents]
            ) / len(model.majority_agents)
        )
    else:
        print(f'{subset} not recognized')


#: Calculate proportion of churlish agents in a Model instance.
def _proportion_churlish(model, subset=None):

    if subset is None:
        return (
            np.sum(
                [a.receiving_strategy == "Churlish" for a in model.agents]
            ) / model.N
        )
    elif subset == 'minority':
        return (
            np.sum(
                [a.receiving_strategy == "Churlish"
                 for a in model.minority_agents]
            ) / len(model.minority_agents)
        )
    elif subset == 'majority':
        return (
            np.sum(
                [a.receiving_strategy == "Churlish"
                 for a in model.majority_agents]
            ) / len(model.majority_agents)
        )
    else:
        print(f'{subset} not recognized')


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

        # Attribute for tracking what the next strategy will be after
        # a teacher/learner interaction.
        self.next_strategy = None

        # Remember who I have interacted with by their index.
        # self.previous_partners = set()

    def maybe_update_strategy(self, teacher,
                              homophily=0.0, learning_beta=1.0):
        '''
        Update either signaling or receiving strategy to match teacher's with
        probability proportional to logistic difference between teacher and
        learner payoffs.
        '''
        if homophily > 0.0:
            raise NotImplementedError(
                'Learning does not yet support nonrandom teacher selection, '
                'i.e. homophily > 0'
            )

        # See if this learner will update.
        diff = teacher.gross_payoff - self.gross_payoff
        update = uniform() < _logistic(diff, scale=learning_beta)

        if update:
            # Which strategy is updated is random.
            strategy_type = choice(['signaling', 'receiving'])
            if strategy_type == 'signaling':
                teacher_strategy = teacher.signaling_strategy
            elif strategy_type == 'receiving':
                teacher_strategy = teacher.receiving_strategy
            else:
                raise RuntimeError('Encountered unknown strategy type!')

            self.next_strategy = {strategy_type: teacher_strategy}

        else:
            self.next_strategy = None
