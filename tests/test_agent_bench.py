from jev_like.agent_bench import NEXT_ACTIONS, SCENARIOS, TOOLS, expected_next, ordered_options


def test_scenarios_have_valid_policy_targets():
    assert len({item.scenario_id for item in SCENARIOS}) == len(SCENARIOS)
    for scenario in SCENARIOS:
        assert scenario.expected_tool in TOOLS
        assert scenario.denial_action in NEXT_ACTIONS
        if scenario.expected_tool == "ask user":
            assert scenario.expected_execute is None
        else:
            assert scenario.expected_execute is not None
        for outcome in scenario.outcomes:
            assert expected_next(outcome) in NEXT_ACTIONS


def test_option_order_is_stable_and_complete():
    first = ordered_options(TOOLS, "case:route")
    second = ordered_options(TOOLS, "case:route")
    assert first == second
    assert sorted(first) == sorted(TOOLS)
