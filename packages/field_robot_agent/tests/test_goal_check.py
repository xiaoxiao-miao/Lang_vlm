from field_robot_agent.agent.goal_check import (
    RuleBasedGoalChecker,
    answer_world_query,
    describe_mentioned_objects,
    is_world_query,
    looks_like_apple_in_drawer,
)
from field_robot_agent.world_model.schema import Relation, build_demo_world


def test_query_vs_manipulation_intent():
    assert is_world_query("苹果在哪里")
    assert is_world_query("现在苹果在哪里")
    assert is_world_query("Where is the apple?")
    assert is_world_query("苹果在桌子上吗")
    assert is_world_query("苹果在桌子 上吗")
    assert is_world_query("Is the apple on the table?")
    assert is_world_query("把苹果放进抽屉了吗")
    assert not is_world_query("把苹果放进抽屉")
    assert looks_like_apple_in_drawer("把苹果放进抽屉")
    assert not looks_like_apple_in_drawer("苹果在哪里")
    assert not looks_like_apple_in_drawer("苹果在桌子上吗")


def test_query_answers_from_world_model():
    world = build_demo_world("苹果在哪里")
    world.relations = [
        Relation(subject="drawer_1", predicate="part_of", object="cabinet_1"),
        Relation(subject="apple_1", predicate="inside", object="drawer_1"),
    ]
    world.objects["drawer_1"].attributes["state"] = "open"
    result = RuleBasedGoalChecker().check("苹果在哪里", world)
    assert result.satisfied
    assert "apple_1 inside drawer_1" in result.explanation
    assert "on table_1" not in describe_mentioned_objects("苹果在哪里", world)


def test_yes_no_query_answers_from_world_not_as_unsatisfied_goal():
    world = build_demo_world("苹果在桌子上吗")
    world.relations = [
        Relation(subject="drawer_1", predicate="part_of", object="cabinet_1"),
        Relation(subject="apple_1", predicate="inside", object="drawer_1"),
    ]
    world.objects["drawer_1"].attributes["state"] = "open"
    result = RuleBasedGoalChecker().check("苹果在桌子上吗", world)
    assert result.satisfied
    assert result.explanation.startswith("No.")
    assert "apple_1 inside drawer_1" in result.explanation
    assert "Yes" not in answer_world_query("苹果在桌子上吗", world)
