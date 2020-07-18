import numpy as np

from collections import Counter
from numpy.testing import assert_array_equal, assert_approx_equal, assert_equal


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

    # Need to update the interaction factors after re-defining agent
    # receiving strategies that determine initial attitudes.
    model._calculate_interaction_factors()

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

    def _update_models(a1, a2):
        for m in [model1, model3, model5]:
            m.agents = [a1, a2]
            m._calculate_interaction_factors()

    # Like/like.
    a1, a2 = (Agent(agent_idx=1), Agent(agent_idx=2))
    a1.attitudes[2] = 1
    a2.attitudes[1] = 1

    _update_models(a1, a2)
    assert model1._dyadic_interaction_factor(a1, a2) == 0.6
    assert model3._dyadic_interaction_factor(a1, a2) == 0.8
    assert model5._dyadic_interaction_factor(a1, a2) == 1.0

    # Like/neutral.
    a1.attitudes[2] = 1
    a2.attitudes[1] = 0
    _update_models(a1, a2)
    assert model1._dyadic_interaction_factor(a1, a2) == 0.55
    assert model3._dyadic_interaction_factor(a1, a2) == 0.65
    assert model5._dyadic_interaction_factor(a1, a2) == 0.75

    a1.attitudes[2] = 1
    a2.attitudes[1] = 0
    _update_models(a1, a2)
    assert model1._dyadic_interaction_factor(a1, a2) == 0.55
    assert model3._dyadic_interaction_factor(a1, a2) == 0.65
    assert model5._dyadic_interaction_factor(a1, a2) == 0.75

    # Like/dislike.
    a1.attitudes[2] = 1
    a2.attitudes[1] = -1
    _update_models(a1, a2)
    assert model1._dyadic_interaction_factor(a1, a2) == 0.5
    assert model3._dyadic_interaction_factor(a1, a2) == 0.5
    assert model5._dyadic_interaction_factor(a1, a2) == 0.5

    a1.attitudes[2] = -1
    a2.attitudes[1] = 1
    _update_models(a1, a2)
    assert model1._dyadic_interaction_factor(a1, a2) == 0.5
    assert model3._dyadic_interaction_factor(a1, a2) == 0.5
    assert model5._dyadic_interaction_factor(a1, a2) == 0.5

    # Neutral/neutral.
    a1.attitudes[2] = 0
    a2.attitudes[1] = 0
    _update_models(a1, a2)
    assert model1._dyadic_interaction_factor(a1, a2) == 0.5
    assert model3._dyadic_interaction_factor(a1, a2) == 0.5
    assert model5._dyadic_interaction_factor(a1, a2) == 0.5

    # Neutral/dislike.
    a1.attitudes[2] = -1
    a2.attitudes[1] = 0
    _update_models(a1, a2)
    assert model1._dyadic_interaction_factor(a1, a2) == 0.45
    assert model3._dyadic_interaction_factor(a1, a2) == 0.35
    assert model5._dyadic_interaction_factor(a1, a2) == 0.25

    a1.attitudes[2] = 0
    a2.attitudes[1] = -1
    _update_models(a1, a2)
    assert model1._dyadic_interaction_factor(a1, a2) == 0.45
    assert model3._dyadic_interaction_factor(a1, a2) == 0.35
    assert model5._dyadic_interaction_factor(a1, a2) == 0.25

    # Dislike/dislike.
    a1.attitudes[2] = -1
    a2.attitudes[1] = -1
    _update_models(a1, a2)
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

    model._calculate_interaction_factors()

    a0_expected = [0.0, 1.4, 1.0, 1.0]
    a1_expected = [1.4, 0.0, 1.0, 0.6]

    a0_expected = np.array(a0_expected) / np.sum(a0_expected)
    a1_expected = np.array(a1_expected) / np.sum(a1_expected)

    # Test three others.
    others = agents[1:]
    assert_array_equal(model._interaction_probs(a0, others), a0_expected)
    others = [a0, a2, a3]
    assert_array_equal(model._interaction_probs(a1, others), a1_expected)

    # Two others not possible with N=4, but that's OK for this test.
    others = [a0, a3]
    a2_expected = np.array([1.0, 0.0, 0.0, 0.8]) / 1.8
    a2_expected = np.array(a2_expected) / np.sum(a2_expected)
    assert_array_equal(model._interaction_probs(a2, others), a2_expected)

    # Need to correctly handle case of one other agent. This is a dumb way
    # to handle this particular case, but we are just making sure it's doing
    # what we expect.
    others = [a1]
    a3_expected = np.array([0.0, 1.0, 0.0, 0.0])
    a3_expected = np.array(a3_expected) / np.sum(a3_expected)
    assert_array_equal(model._interaction_probs(a3, others), a3_expected)

    # Test case of perfect homophily and a_ij = a_ji = -1 for all possible
    # interaction partners.
    model, agents = _setup_model_agents(homophily=0.5)

    a0 = agents[0]
    a1 = agents[1]
    a2 = agents[2]
    a3 = agents[3]

    a0.attitudes = [-1, -1, -1, -1]
    a1.attitudes = [-1, 0, 1, -1]
    a2.attitudes = [-1, 1, 0, 0]
    a3.attitudes = [-1, 1, 1, 0]
    model._calculate_interaction_factors()

    others = [a1, a2, a3]

    n_trials = 1000
    results = np.zeros(n_trials)

    # Check over many instances that the other agents are chosen equally often
    # and that the focal agent a0 never has probability > 0 of interacting
    # with itself, which is forbidden.
    for t_idx in range(n_trials):
        probs = model._interaction_probs(a0, others)
        selected_agent_idx = np.where(probs == 1.0)
        assert len(selected_agent_idx) == 1
        selected_agent_idx = selected_agent_idx[0]
        assert selected_agent_idx != 0
        results[t_idx] = selected_agent_idx

    counts = Counter(results)
    for val in counts.values():
        assert_approx_equal(val/n_trials, 1/3, 1)


def test_make_dyads():

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

    # Need to re-calculate interaction factors after setting attitudes.
    model._calculate_interaction_factors()

    a0_expected = [0.0, 1.4, 1.0, 1.0]
    a1_expected = [1.4, 0.0, 1.0, 0.6]
    a2_expected = [1.0, 1.0, 0.0, 0.8]
    a3_expected = [1.0, 0.6, 0.8, 0.0]

    a0_expected = np.array(a0_expected) / np.sum(a0_expected)
    a1_expected = np.array(a1_expected) / np.sum(a1_expected)
    a2_expected = np.array(a2_expected) / np.sum(a2_expected)
    a3_expected = np.array(a3_expected) / np.sum(a3_expected)

    # With four agents we can easily calculate the probability any given pair
    # i and j interact. It is Pr(i interacts with j) = Pr(i chooses j) +
    # Pr(j chooses i) + Pr(k chooses l) + Pr(l chooses k) = Pr(k interacts with l).
    amat = np.array([a0_expected, a1_expected, a2_expected, a3_expected])
    nmat = amat / amat.sum()

    expected_frequencies = {
        (0, 1): nmat[0, 1] + nmat[1, 0] + nmat[2, 3] + nmat[3, 2],
        (0, 2): nmat[0, 2] + nmat[2, 0] + nmat[1, 3] + nmat[3, 1],
        (0, 3): nmat[0, 3] + nmat[3, 0] + nmat[1, 2] + nmat[2, 1],
        (1, 2): nmat[1, 2] + nmat[2, 1] + nmat[0, 3] + nmat[3, 0],
        (1, 3): nmat[1, 3] + nmat[3, 1] + nmat[0, 2] + nmat[2, 0],
        (2, 3): nmat[2, 3] + nmat[3, 2] + nmat[0, 1] + nmat[1, 0]
    }

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

    for key in calculated_frequencies.keys():
        assert_approx_equal(
            calculated_frequencies[key], expected_frequencies[key], 2
        )


##
# Testing agent-focused learning.
#

def test_maybe_update_strategy():

    a0 = Agent()
    a1 = Agent()

    a0.gross_payoff = 1.0
    a1.gross_payoff = 3.0

    a0.index = 0
    a1.index = 1

    model = Model()
    model.agents = [a0, a1]

    # Each agent will both teach and learn from each other, so the expected
    # frequency is f(teacher - learner).
    a0_expected_freq = 0.8807970779778824
    a1_expected_freq = 0.11920292202211755

    n_trials = 10000
    a0_changed = np.zeros(n_trials, bool)
    a1_changed = np.zeros(n_trials, bool)

    for idx in range(n_trials):
        # a0 & a1 are references to the same model.agents.
        a0.signaling_strategy = 'Overt'
        a0.receiving_strategy = 'Generous'

        a1.signaling_strategy = 'Covert'
        a1.receiving_strategy = 'Churlish'

        model._social_learning()

        a0_changed[idx] = (
            a0.signaling_strategy != 'Overt' or
            a0.receiving_strategy != 'Generous'
        )

        a1_changed[idx] = (
            a1.signaling_strategy != 'Covert' or
            a1.receiving_strategy != 'Churlish'
        )

    assert_approx_equal(a0_changed.sum(), n_trials * a0_expected_freq, 2)
    assert_approx_equal(a1_changed.sum(), n_trials * a1_expected_freq, 2)


def _run_maybe_update_trials(model, teacher, learner, n_trials):

    changed_strategies = np.zeros(n_trials, dtype=bool)

    for idx in range(n_trials):
        teacher.signaling_strategy = 'Overt'
        learner.signaling_strategy = 'Covert'

        teacher.receiving_strategy = 'Generous'
        learner.receiving_strategy = 'Churlish'

        learner.maybe_update_strategy(teacher)

        changed_strategies[idx] = (
            learner.signaling_strategy != 'Covert' or
            learner.receiving_strategy != 'Churlish'
        )

    return changed_strategies


def test_logistic():
    '''
    Check that logistic function gives expected probabilities.
    '''
    from id_signaling.model import _logistic

    assert_equal(_logistic(1), 0.7310585786300049)
    assert_equal(_logistic(2, scale=0.5), 0.7310585786300049)
    assert_equal(_logistic(0.5, scale=2), 0.7310585786300049)

    assert_equal(_logistic(2), 0.8807970779778823)
    assert_equal(_logistic(1, scale=2), 0.8807970779778823)

    assert_approx_equal(_logistic(-2), 1 - 0.8807970779778823, significant=9)
    assert_approx_equal(_logistic(-1, scale=2), 1 - 0.8807970779778823, significant=9)

    assert_equal(_logistic(3), 0.9525741268224334)
    assert_equal(_logistic(1, scale=3), 0.9525741268224334)

    assert_approx_equal(_logistic(-3), 1 - 0.9525741268224334, significant=9)
    assert_approx_equal(_logistic(-1, scale=3), 1 - 0.9525741268224334, significant=9)


def test_minority_learning():
    '''
    In the current version of the model, minorities should only learn from other minorities.
    '''

    # First, create model and four agents, two "minority" two not.
    min0 = Agent(minority=True)
    min1 = Agent(minority=True)
    maj = Agent()
    # maj1 = Agent()

    # Next, set up a situation where one minority agent will always adopt the
    # other minority agent's strategy, and the other minority agent will never
    # adopt the other minority's opinion. If, however, the minority agents
    # interact with majority agents the opposite will happen. So, this
    # checks that minority only ever interact with minority.

    # min0 should always adopt min1's strategies, and never keep their own
    # unless they interact with maj1 in the minority. Similarly, min1 will
    # should always keep its strategies unless it interacts with maj, when
    # it would always switch strategies.
    min0.gross_payoff = 1.0
    min1.gross_payoff = 3.0
    maj.gross_payoff = 4.0

    min0.receiving_strategy = 'Generous'
    min0.signaling_strategy = 'Covert'

    min1.receiving_strategy = 'Churlish'
    min1.signaling_strategy = 'Overt'

    maj.receiving_strategy = 'Generous'
    maj.signaling_strategy = 'Covert'

    model = Model(learning_beta=1e6)
    # Normally this is set only if a minority_frac is given and not set
    # directly. But we circumvent this convenience for this test.
    model.minority_test = True
    model.agents = [min0, min1, maj]

    for _ in range(10000):

        model._social_learning()

        # min0 must have changed one strategy.
        assert (
            min0.receiving_strategy == 'Churlish' or
            min0.signaling_strategy == 'Overt'
        )
        # min1 should have not changed strategy.
        assert (
            min1.receiving_strategy == 'Churlish' and
            min1.signaling_strategy == 'Overt'
        )
        # maj should never change strategy since its payoffs are superior;
        # sort of unrelated, but why not.
        assert (
            maj.receiving_strategy == 'Generous' and
            maj.signaling_strategy == 'Covert'
        )
