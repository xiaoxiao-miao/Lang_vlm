"""MockPolicy unit tests."""

from robot_agent.policies.mock import MockPolicy


def test_mock_policy_action_floats():
    policy = MockPolicy(horizon=6)
    policy.set_robot_state_keys(
        ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]
    )
    actions = policy.get_actions(
        {"observation.state": [0.0] * 6},
        "抓取桌面红色物体",
    )
    assert len(actions) == 6
    assert actions[-1]["grasp"] == 1.0
    assert actions[-1]["gripper"] >= 0.7
    for tick in actions:
        for key, value in tick.items():
            assert isinstance(value, float), f"{key}={value!r}"
