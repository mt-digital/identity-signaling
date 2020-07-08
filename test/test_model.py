import numpy as np

from numpy.testing import assert_array_equal, assert_approx_equal

from id_signaling.model import Agent, Model


def test_minorities():
    'Check that minorities experiment is properly initialized and payoffs are as expected.'

    _test_minorities(3, 1, 0.1)
    _test_minorities(3, 1, 0.25)
    _test_minorities(3, 1, 0.75)

    _test_minorities(9, 4, 0.1)
    _test_minorities(9, 4, 0.25)
    _test_minorities(9, 4, 0.75)


def _test_minorities(K, M, minority_trait_frac, N=100):

    # Set up new model with given settings.
    model = Model(N=N, K=K, n_minmaj_traits=M,
                  minority_trait_frac=minority_trait_frac)

    # Simplest thing to test is the setup. Do we really have a minority and
    # majority where the number of agents with the minority/majority trait
    # are as expected?

    agents = model.agents

    minority = [a for a in agents if (a.traits[0:M] == [1]*M).all()]
    majority = [a for a in agents if (a.traits[0:M] == [-1]*M).all()]
    assert len(minority) == np.floor(minority_trait_frac * N)
    assert len(majority) == np.floor((1 - minority_trait_frac) * N)


def test_similarity_threshold():
    'Check that payoffs and matching probabilities are as expected for different similarity thresholds.'

    # Initialize model.
    m = Model(similarity_threshold=0.5)

    # Set up some agents to be similar or dissimlar.
    a1 = Agent(K=3, agent_idx=0)
    a2 = Agent(K=3, agent_idx=1)
    a3 = Agent(K=3, agent_idx=2)

    a1.traits = np.array([1, 1, -1])
    a2.traits = np.array([1, 1, 1])
    a3.traits = np.array([1, -1, 1])

    m.agents = [a1, a2, a3]
    m._init_similar_matrix()

    assert m._are_similar(a1, a2)
    assert not m._are_similar(a1, a3)
    assert m._are_similar(a2, a3)

    m = Model(similarity_threshold=0.1)
    m.agents = [a1, a2, a3]
    m._init_similar_matrix()
    assert m._are_similar(a1, a2)
    assert m._are_similar(a1, a3)
    assert m._are_similar(a2, a3)

    m = Model(similarity_threshold=0.9)
    m.agents = [a1, a2, a3]
    m._init_similar_matrix()
    assert not m._are_similar(a1, a2)
    assert not m._are_similar(a1, a3)
    assert not m._are_similar(a2, a3)
    assert m._are_similar(a1, a1)
    assert m._are_similar(a2, a2)
    assert m._are_similar(a3, a3)

    m = Model(similarity_threshold=0.5)
    # Set up some agents to be similar or dissimlar.
    a1 = Agent(K=10, agent_idx=0)
    a2 = Agent(K=10, agent_idx=1)
    a3 = Agent(K=10, agent_idx=2)
    a1.traits = np.array([1]*5 + [-1]*5)
    a2.traits = np.array([1]*10)
    a3.traits = np.array([-1]*4 + [1]*6)
    m.agents = [a1, a2, a3]
    m._init_similar_matrix()

    assert m._are_similar(a1, a2)
    assert m._are_similar(a2, a3)
    assert not m._are_similar(a1, a3)


def test_invasion_setup():
    'Check that there are correct number of invading and established populations.'

    _test_invasion_setup()
    _test_invasion_setup(0.1, 0.9)
    _test_invasion_setup(0.9, 0.1)
    _test_invasion_setup(0.1, 0.5)
    _test_invasion_setup(0.5, 0.1)


def _test_invasion_setup(init_cov=0.5, init_ch=0.5, N=100):

    m = Model(N=N, initial_prop_covert=init_cov, initial_prop_churlish=init_ch)

    cov = [a for a in m.agents if a.signaling_strategy == 'Covert']
    chur = [a for a in m.agents if a.receiving_strategy == 'Churlish']

    assert len(cov) == np.floor(init_cov * N)
    assert len(chur) == np.floor(init_ch * N)

##
# Expected payoffs on interaction. Using results from test above we know the
# attitudes of the agents after they've signaled and received to one another
# and, in the absence of evolution, interacted for one thousand rounds so
# payoffs will be approximately equal to the expected payoff calculated by
# hand.
#
def test_calculate_payoff():
    'First just check that payoffs are as expected'
    model = Model(similarity_benefit=0.5,
                  one_dislike_penalty=0.25,
                  two_dislike_penalty=0.15)

    a0 = Agent(agent_idx=0)
    a1 = Agent(agent_idx=1)

    # Like/like. Only similar possible. Only one possible configuration.
    a0.traits = np.array([1, 1, -1])
    a1.traits = np.array([1, -1, -1])
    model.agents = [a0, a1]
    model._init_similar_matrix()

    a0.attitudes = np.array([0, 1])
    a1.attitudes = np.array([1, -1])
    assert model._calculate_payoff(a0, a1) == 1 + 0.5

    # Like/neutral. Only similar possible.
    a0.attitudes = np.array([0, 1])
    a1.attitudes = np.array([0, -1])
    assert model._calculate_payoff(a0, a1) == 1 + 0.5

    a0.attitudes = np.array([0, 0])
    a1.attitudes = np.array([1, -1])
    assert model._calculate_payoff(a0, a1) == 1 + 0.5

    # Like/dislike, similar.
    a0.traits = np.array([1, 1, -1])
    a1.traits = np.array([1, -1, -1])
    model._init_similar_matrix()

    a0.attitudes = np.array([0, 1])
    a1.attitudes = np.array([-1, -1])
    assert model._calculate_payoff(a0, a1) == 1 + 0.5 - 0.25

    a0.attitudes = np.array([0, -1])
    a1.attitudes = np.array([1, -1])
    assert model._calculate_payoff(a0, a1) == 1 + 0.5 - 0.25

    # Like/dislike, dissimilar. -- Impossible?
    a0.traits = np.array([1, 1, -1])
    a1.traits = np.array([-1, -1, -1])

    # Need to re-initialize similarity matrix whenever traits updated.
    model._init_similar_matrix()
    print(model.similar_matrix)

    a0.attitudes = np.array([0, 1])
    a1.attitudes = np.array([-1, -1])
    assert model._calculate_payoff(a0, a1) == 1 - 0.25

    a0.attitudes = np.array([0, -1])
    a1.attitudes = np.array([1, -1])
    assert model._calculate_payoff(a0, a1) == 1 - 0.25

    # Neutral/neutral, similar.
    a0.traits = np.array([1, 1, -1])
    a1.traits = np.array([1, -1, -1])

    model._init_similar_matrix()

    a0.attitudes = np.array([0, 0])
    a1.attitudes = np.array([0, -1])
    assert model._calculate_payoff(a0, a1) == 1 + 0.5

    # Neutral/neutral, dissimilar. Only one configuration.
    a0.traits = np.array([1, 1, -1])
    a1.traits = np.array([-1, -1, -1])

    model._init_similar_matrix()

    a0.attitudes = np.array([0, 0])
    a1.attitudes = np.array([0, -1])
    assert model._calculate_payoff(a0, a1) == 1

    # Neutral/dislike, similar.
    a0.traits = np.array([1, 1, -1])
    a1.traits = np.array([1, -1, -1])

    model._init_similar_matrix()

    a0.attitudes = np.array([0, 0])
    a1.attitudes = np.array([-1, -1])
    assert model._calculate_payoff(a0, a1) == 1 + 0.5 - 0.25

    a0.attitudes = np.array([0, -1])
    a1.attitudes = np.array([0, -1])
    assert model._calculate_payoff(a0, a1) == 1 + 0.5 - 0.25

    # Neutral/dislike, dissimilar.
    a0.traits = np.array([1, 1, -1])
    a1.traits = np.array([-1, -1, -1])

    model._init_similar_matrix()

    a0.attitudes = np.array([0, 0])
    a1.attitudes = np.array([-1, -1])
    assert model._calculate_payoff(a0, a1) == 1 - 0.25

    a0.attitudes = np.array([0, -1])
    a1.attitudes = np.array([0, -1])
    assert model._calculate_payoff(a0, a1) == 1 - 0.25

    # Dislike/dislike, similar. Only one configuration.
    a0.traits = np.array([1, 1, -1])
    a1.traits = np.array([1, -1, -1])

    model._init_similar_matrix()

    a0.attitudes = np.array([0, -1])
    a1.attitudes = np.array([-1, -1])
    assert model._calculate_payoff(a0, a1) == 1 + 0.5 - 0.25 - 0.15

    # Dislike/dislike, dissimilar. Only one configuration.
    a0.traits = np.array([1, 1, -1])
    a1.traits = np.array([-1, -1, -1])

    model._init_similar_matrix()

    a0.attitudes = np.array([0, -1])
    a1.attitudes = np.array([-1, -1])
    assert model._calculate_payoff(a0, a1) == 1 - 0.25 - 0.15


##
# Signaling and receiving. Need to ensure generous/churlish receivers act as
# we expect them to depending on covert/overt signaling from either similar
# or different others.
#
def test_signal_and_receive():
    # We just want to know that different strategies influence one another
    # as expected, so don't care about payoffs, percentages of receivers,
    # etc.
    model, agents = _setup_model_agents(prob_overt_receiving=1.0,
                                        prob_covert_receiving=1.0)

    # Check now that generous and churlish receiving interacted as expected
    # with signaling strategies and given traits.

    # 0 is similar to 1, so doesn't matter if 1 is covert or overt, 0 will
    # like 1; 2 is a dissimilar overt signaler, so 0 dislikes 2; 3 is also
    # dissimilar but is a covert signaler, so 0 stays neutral (0 is generous).
    assert_array_equal(agents[0].attitudes, np.array([0, 1, -1, 0]))
    # Vice versa for 1-to-0 relationship and same for 1-2 relationship;
    # 3 is dissimilar but a covert signaler, so 1 stays neutral (1 generous).
    assert_array_equal(agents[1].attitudes, np.array([1, 0, 1, 0]))
    # 2 is dissimilar from 0 who is an overt signaler; 2 similar to 1 and 3.
    assert_array_equal(agents[2].attitudes, np.array([-1, 1, -1, 1]))
    # 3 is dissimilar from agent 0, an overt signaller; also dissimilar from
    # 1 who is a covert signaler - as a churlish receiver 3 keeps a dislike
    # attitude towards 1; 3 similar to 2.
    assert_array_equal(agents[3].attitudes, np.array([-1, -1, 1, -1]))


def test_expected_payoffs():
    '''
    Check that the gross payoffs for each agent match the expected gross
    payoffs in our simple test case. <π_i> = ∑ p(i,j)π(i,j).
    '''

    n_rounds = 10000
    model, agents = _setup_model_agents(
        prob_overt_receiving=1.0, prob_covert_receiving=1.0,
        homophily=0.0, n_rounds=n_rounds
    )

    # Write out π(i,j) interact calculated by hand.
    pi01 = 1 + 0.25; pi02 = 1 - 0.25 - 0.25; pi03 = 1 - 0.25
    pi10 = pi01    ; pi12 = 1 + 0.25       ; pi13 = 1 - 0.25
    pi20 = pi02    ; pi21 = pi12           ; pi23 = 1 + 0.25
    pi30 = pi03    ; pi31 = pi13           ; pi32 = pi23

    # Expected payoffs. 1/2 for w=0, 1/3 for interacting with other agents
    # approx 1/3 the time. There may be a combinatorial correction since
    # pairing two agents changes conditional prob of other agent pairings.
    pi0 = (1/3) * (pi01 + pi02 + pi03) * n_rounds
    pi1 = (1/3) * (pi10 + pi12 + pi13) * n_rounds
    pi2 = (1/3) * (pi20 + pi21 + pi23) * n_rounds
    pi3 = (1/3) * (pi30 + pi31 + pi32) * n_rounds

    assert_approx_equal(model.agents[0].gross_payoff, pi0, 2)
    assert_approx_equal(model.agents[1].gross_payoff, pi1, 2)
    assert_approx_equal(model.agents[2].gross_payoff, pi2, 2)
    assert_approx_equal(model.agents[3].gross_payoff, pi3, 2)


def _setup_model_agents(**model_kwargs):

    model = Model(N=4, **model_kwargs)

    model.agents = [
        Agent(N=4, K=3, agent_idx=0,
              receiving_strategy="Generous", signaling_strategy="Overt"),
        Agent(N=4, K=3, agent_idx=1, receiving_strategy="Generous",
              signaling_strategy="Covert"),
        Agent(N=4, K=3, agent_idx=2, receiving_strategy="Churlish",
              signaling_strategy="Overt"),
        Agent(N=4, K=3, agent_idx=3, receiving_strategy="Churlish",
              signaling_strategy="Covert")
    ]

    agents = model.agents

    # Similar agents then should be (0,1), (1,2), and (2,3)
    agents[0].traits = np.array([-1, -1, -1])
    agents[1].traits = np.array([-1, 1, -1])
    agents[2].traits = np.array([-1, 1, 1])
    agents[3].traits = np.array([1, 1, 1])

    # Check that generous and churlish agents properly initialized.
    assert_array_equal(agents[0].attitudes, np.array([0, 0, 0, 0]))
    assert_array_equal(agents[1].attitudes, np.array([0, 0, 0, 0]))
    assert_array_equal(agents[2].attitudes, np.array([-1, -1, -1, -1]))
    assert_array_equal(agents[3].attitudes, np.array([-1, -1, -1, -1]))

    # Have to update similarity matrix using call to private method usually
    # only called in Model constructor.
    model._init_similar_matrix()

    # Because we set overt and covert signals to always be received,
    # running the model just one timestep should update all attitudes.
    model.run(1)

    return model, agents


##
# Calculating dyadic interaction probabilities.
#
def test_dyadic_interaction_factors():
    '''
    Dyadic interaction probabilities should match expected values calculated

    '''
    # Check that dyadic interaction probabilities are as expected for each
    # pair of agents that will be passed in above
    model1 = Model(homophily=0.1)
    model3 = Model(homophily=0.3)
    model5 = Model(homophily=0.5)

    # Like/like.
    a1, a2 = (Agent(agent_idx=1), Agent(agent_idx=2))
    a1.attitudes[2] = 1
    a2.attitudes[1] = 1
    assert model1._dyadic_interaction_factor(a1, a2) == 0.6
    assert model3._dyadic_interaction_factor(a1, a2) == 0.8
    assert model5._dyadic_interaction_factor(a1, a2) == 1.0

    # Like/neutral.
    a1.attitudes[2] = 1
    a2.attitudes[1] = 0
    assert model1._dyadic_interaction_factor(a1, a2) == 0.55
    assert model3._dyadic_interaction_factor(a1, a2) == 0.65
    assert model5._dyadic_interaction_factor(a1, a2) == 0.75

    a1.attitudes[2] = 1
    a2.attitudes[1] = 0
    assert model1._dyadic_interaction_factor(a1, a2) == 0.55
    assert model3._dyadic_interaction_factor(a1, a2) == 0.65
    assert model5._dyadic_interaction_factor(a1, a2) == 0.75

    # Like/dislike.
    a1.attitudes[2] = 1
    a2.attitudes[1] = -1
    assert model1._dyadic_interaction_factor(a1, a2) == 0.5
    assert model3._dyadic_interaction_factor(a1, a2) == 0.5
    assert model5._dyadic_interaction_factor(a1, a2) == 0.5

    a1.attitudes[2] = -1
    a2.attitudes[1] = 1
    assert model1._dyadic_interaction_factor(a1, a2) == 0.5
    assert model3._dyadic_interaction_factor(a1, a2) == 0.5
    assert model5._dyadic_interaction_factor(a1, a2) == 0.5

    # Neutral/neutral.
    a1.attitudes[2] = 0
    a2.attitudes[1] = 0
    assert model1._dyadic_interaction_factor(a1, a2) == 0.5
    assert model3._dyadic_interaction_factor(a1, a2) == 0.5
    assert model5._dyadic_interaction_factor(a1, a2) == 0.5

    # Neutral/dislike.
    a1.attitudes[2] = -1
    a2.attitudes[1] = 0
    assert model1._dyadic_interaction_factor(a1, a2) == 0.45
    assert model3._dyadic_interaction_factor(a1, a2) == 0.35
    assert model5._dyadic_interaction_factor(a1, a2) == 0.25

    a1.attitudes[2] = 0
    a2.attitudes[1] = -1
    assert model1._dyadic_interaction_factor(a1, a2) == 0.45
    assert model3._dyadic_interaction_factor(a1, a2) == 0.35
    assert model5._dyadic_interaction_factor(a1, a2) == 0.25

    # Dislike/dislike.
    a1.attitudes[2] = -1
    a2.attitudes[1] = -1
    assert model1._dyadic_interaction_factor(a1, a2) == 0.4
    assert model3._dyadic_interaction_factor(a1, a2) == 0.2
    assert model5._dyadic_interaction_factor(a1, a2) == 0.0


def test_interaction_probs():
    '''
    Calculate interaction probability vector for all other agents and itself, which should always be 0.
    '''
    model, agents = _setup_model_agents(homophily=0.2)

    a0 = agents[0]
    a1 = agents[1]
    a2 = agents[2]
    a3 = agents[3]

    a0.attitudes = [0, 1, 0, -1]
    a1.attitudes = [1, 0, -1, -1]
    a2.attitudes = [0, 1, 0, 0]
    a3.attitudes = [1, -1, -1, 0]

    a0_expected = [0.0, 1.4, 1.0, 1.0]
    a1_expected = [1.4, 0.0, 1.0, 0.6]
    a3_expected = [1.0, 0.6, 0.8, 0.0]

    a0_expected = np.array(a0_expected) / np.sum(a0_expected)
    a1_expected = np.array(a1_expected) / np.sum(a1_expected)
    a2_expected = np.array(a2_expected) / np.sum(a2_expected)
    a3_expected = np.array(a3_expected) / np.sum(a3_expected)

    # Test three others.
    others = agents[1:]
    assert_array_equal(model._interaction_probs(a0), a0_expected, others)
    others = [a0, a2, a3]
    assert_array_equal(model._interaction_probs(a1), a1_expected, others)

    # Two others not possible with N=4, but that's OK for this test.
    others = [a0, a2]
    a2_expected = np.array([1.0, 0.0, 0.0, 0.8]) / 1.8
    assert_array_equal(model._interaction_probs(a2), a2_expected, others)

    # Need to correctly handle case of one other agent. This is a dumb way
    # to handle this particular case, but we are just making sure it's doing
    # what we expect.
    others = [a1]
    a3_expected = np.array([0.0, 1.0, 0.0, 0.0])
    assert_array_equal(model._interaction_probs(a3), a3_expected, others)


def test_make_dyads():

    from collections import Counter

    model, agents = _setup_model_agents(homophily=0.2)

    # Check that the correct number of pairs are being made. Check that the
    # frequency of pairings is as expected.

    a0 = agents[0]
    a1 = agents[1]
    a2 = agents[2]
    a3 = agents[3]

    a0.attitudes = [0, 1, 0, -1]
    a1.attitudes = [1, 0, -1, -1]
    a2.attitudes = [0, 1, 0, 0]
    a3.attitudes = [1, -1, -1, 0]

    a0_expected = [0.0, 1.4, 1.0, 1.0]
    a1_expected = [1.4, 0.0, 1.0, 0.6]
    a2_expected = [1.0, 1.0, 0.0, 0.8]
    a3_expected = [1.0, 0.6, 0.8, 0.0]

    a0_expected = np.array(a0_expected) / np.sum(a0_expected)
    a1_expected = np.array(a1_expected) / np.sum(a1_expected)
    a2_expected = np.array(a2_expected) / np.sum(a2_expected)
    a3_expected = np.array(a3_expected) / np.sum(a3_expected)

    # I am expecting that, since who picks their partner first is random that
    # the frequency of pairings will average out to be equal to the probability
    # of interaction of each pair.
    expected_frequencies = {
        (0, 1): a0_expected[1], (0, 2): a0_expected[2],
        (0, 3): a0_expected[3], (1, 2): a1_expected[2],
        (1, 3): a1_expected[3], (2, 3): a2_expected[3]
    }

    all_dyads = []  # model._make_dyads()
    n_trials = 10000
    all_dyads = [set(a.index for a in dyad) for dyad in model._make_dyads()]
    print(all_dyads)
    for ii in range(n_trials):
        # XXX super inefficient, would be good to clean up.
        all_dyads = np.append(
            all_dyads,
            [set(a.index for a in dyad) for dyad in model._make_dyads()]
        )

    # Sort each pairing to get the keys in expected_frequencies above.
    all_dyads = [tuple(np.sort(np.array(list(dyad)))) for dyad in all_dyads]

    # First step: count number of instances of each series.
    calculated_frequencies = Counter(all_dyads)

    # Next: calculate frequency by dividing by number of trials.
    calculated_frequencies = {
        k: v / n_trials for k, v in calculated_frequencies.items()
    }

    assert calculated_frequencies == expected_frequencies
