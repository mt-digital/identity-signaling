import numpy as np
import pytest

from numpy.testing import assert_array_equal, assert_approx_equal

from id_signaling.model import Agent, Model


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

    a0.attitudes = np.array([0, 1])
    a1.attitudes = np.array([-1, -1])
    assert model._calculate_payoff(a0, a1) == 1 + 0.5 - 0.25

    a0.attitudes = np.array([0, -1])
    a1.attitudes = np.array([1, -1])
    assert model._calculate_payoff(a0, a1) == 1 + 0.5 - 0.25

    # Like/dislike, dissimilar. -- Impossible?
    a0.traits = np.array([1, 1, -1])
    a1.traits = np.array([-1, -1, -1])

    a0.attitudes = np.array([0, 1])
    a1.attitudes = np.array([-1, -1])
    assert model._calculate_payoff(a0, a1) == 1 - 0.25

    a0.attitudes = np.array([0, -1])
    a1.attitudes = np.array([1, -1])
    assert model._calculate_payoff(a0, a1) == 1 - 0.25

    # Neutral/neutral, similar.
    a0.traits = np.array([1, 1, -1])
    a1.traits = np.array([1, -1, -1])

    a0.attitudes = np.array([0, 0])
    a1.attitudes = np.array([0, -1])
    assert model._calculate_payoff(a0, a1) == 1 + 0.5

    # Neutral/neutral, dissimilar. Only one configuration.
    a0.traits = np.array([1, 1, -1])
    a1.traits = np.array([-1, -1, -1])

    a0.attitudes = np.array([0, 0])
    a1.attitudes = np.array([0, -1])
    assert model._calculate_payoff(a0, a1) == 1

    # Neutral/dislike, similar.
    a0.traits = np.array([1, 1, -1])
    a1.traits = np.array([1, -1, -1])

    a0.attitudes = np.array([0, 0])
    a1.attitudes = np.array([-1, -1])
    assert model._calculate_payoff(a0, a1) == 1 + 0.5 - 0.25

    a0.attitudes = np.array([0, -1])
    a1.attitudes = np.array([0, -1])
    assert model._calculate_payoff(a0, a1) == 1 + 0.5 - 0.25

    # Neutral/dislike, dissimilar.
    a0.traits = np.array([1, 1, -1])
    a1.traits = np.array([-1, -1, -1])

    a0.attitudes = np.array([0, 0])
    a1.attitudes = np.array([-1, -1])
    assert model._calculate_payoff(a0, a1) == 1 - 0.25

    a0.attitudes = np.array([0, -1])
    a1.attitudes = np.array([0, -1])
    assert model._calculate_payoff(a0, a1) == 1 - 0.25

    # Dislike/dislike, similar. Only one configuration.
    a0.traits = np.array([1, 1, -1])
    a1.traits = np.array([1, -1, -1])

    a0.attitudes = np.array([0, -1])
    a1.attitudes = np.array([-1, -1])
    assert model._calculate_payoff(a0, a1) == 1 + 0.5 - 0.25 - 0.15

    # Dislike/dislike, dissimilar. Only one configuration.
    a0.traits = np.array([1, 1, -1])
    a1.traits = np.array([-1, -1, -1])

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
    pi0 = 0.5 * (1/3) * (pi01 + pi02 + pi03) * n_rounds
    pi1 = 0.5 * (1/3) * (pi10 + pi12 + pi13) * n_rounds
    pi2 = 0.5 * (1/3) * (pi20 + pi21 + pi23) * n_rounds
    pi3 = 0.5 * (1/3) * (pi30 + pi31 + pi32) * n_rounds

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

    # Because we set overt and covert signals to always be received,
    # running the model just one timestep should update all attitudes.
    model.run(1)

    return model, agents


##
# Calculating dyadic interaction probabilities.
#
def test_dyadic_interaction_probs():
# def test_dyadic_interaction_probs():
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
    assert model1._dyadic_interaction_prob(a1, a2) == 0.6
    assert model3._dyadic_interaction_prob(a1, a2) == 0.8
    assert model5._dyadic_interaction_prob(a1, a2) == 1.0

    # Like/neutral.
    a1.attitudes[2] = 1
    a2.attitudes[1] = 0
    assert model1._dyadic_interaction_prob(a1, a2) == 0.55
    assert model3._dyadic_interaction_prob(a1, a2) == 0.65
    assert model5._dyadic_interaction_prob(a1, a2) == 0.75

    a1.attitudes[2] = 1
    a2.attitudes[1] = 0
    assert model1._dyadic_interaction_prob(a1, a2) == 0.55
    assert model3._dyadic_interaction_prob(a1, a2) == 0.65
    assert model5._dyadic_interaction_prob(a1, a2) == 0.75

    # Like/dislike.
    a1.attitudes[2] = 1
    a2.attitudes[1] = -1
    assert model1._dyadic_interaction_prob(a1, a2) == 0.5
    assert model3._dyadic_interaction_prob(a1, a2) == 0.5
    assert model5._dyadic_interaction_prob(a1, a2) == 0.5

    a1.attitudes[2] = -1
    a2.attitudes[1] = 1
    assert model1._dyadic_interaction_prob(a1, a2) == 0.5
    assert model3._dyadic_interaction_prob(a1, a2) == 0.5
    assert model5._dyadic_interaction_prob(a1, a2) == 0.5

    # Neutral/neutral.
    a1.attitudes[2] = 0
    a2.attitudes[1] = 0
    assert model1._dyadic_interaction_prob(a1, a2) == 0.5
    assert model3._dyadic_interaction_prob(a1, a2) == 0.5
    assert model5._dyadic_interaction_prob(a1, a2) == 0.5

    # Neutral/dislike.
    a1.attitudes[2] = -1
    a2.attitudes[1] = 0
    assert model1._dyadic_interaction_prob(a1, a2) == 0.45
    assert model3._dyadic_interaction_prob(a1, a2) == 0.35
    assert model5._dyadic_interaction_prob(a1, a2) == 0.25

    a1.attitudes[2] = 0
    a2.attitudes[1] = -1
    assert model1._dyadic_interaction_prob(a1, a2) == 0.45
    assert model3._dyadic_interaction_prob(a1, a2) == 0.35
    assert model5._dyadic_interaction_prob(a1, a2) == 0.25

    # Dislike/dislike.
    a1.attitudes[2] = -1
    a2.attitudes[1] = -1
    assert model1._dyadic_interaction_prob(a1, a2) == 0.4
    assert model3._dyadic_interaction_prob(a1, a2) == 0.2
    assert model5._dyadic_interaction_prob(a1, a2) == 0.0
