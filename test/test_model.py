import numpy as np
import pytest

from numpy.testing import assert_array_equal

from id_signaling.model import Agent, Model


##
# Signaling and receiving. Need to ensure generous/churlish receivers act as
# we expect them to depending on covert/overt signaling from either similar
# or different others.
#
def test_signal_and_receive():
    # We just want to know that different strategies influence one another
    # as expected, so don't care about payoffs, percentages of receivers,
    # etc.
    model = Model(N=4, prop_overt=1.0, prop_covert=1.0)

    model.agents = [
        Agent(N=4, K=3, agent_idx=0, receiving_strategy="Generous", signaling_strategy="Overt"),
        Agent(N=4, K=3, agent_idx=1, receiving_strategy="Generous", signaling_strategy="Covert"),
        Agent(N=4, K=3, agent_idx=2, receiving_strategy="Churlish", signaling_strategy="Overt"),
        Agent(N=4, K=3, agent_idx=3, receiving_strategy="Churlish", signaling_strategy="Covert")
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
