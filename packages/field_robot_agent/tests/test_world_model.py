"""WorldModelManager event application."""

from field_robot_agent.world_model.events import AttributeChange, RelationChange, WorldEvent
from field_robot_agent.world_model.manager import PROTECT_THRESHOLD, WorldModelManager
from field_robot_agent.world_model.schema import ObjectState, build_demo_world


def test_closed_to_open_increments_revision():
    world = build_demo_world("demo")
    assert world.objects["drawer_1"].attributes["state"] == "closed"
    start = world.revision
    event = WorldEvent(
        event_type="drawer_opened",
        description="open",
        attribute_changes=[
            AttributeChange(
                entity_id="drawer_1",
                attribute="state",
                old_value="closed",
                new_value="open",
                confidence=1.0,
            )
        ],
    )
    updated = WorldModelManager().apply_event(world, event)
    assert updated.objects["drawer_1"].attributes["state"] == "open"
    assert updated.revision == start + 1
    assert world.objects["drawer_1"].attributes["state"] == "closed"


def test_relation_add_and_remove():
    world = build_demo_world("demo")
    manager = WorldModelManager()
    added = manager.apply_event(
        world,
        WorldEvent(
            event_type="placed",
            description="add inside",
            relation_changes=[
                RelationChange(
                    action="add",
                    subject="apple_1",
                    predicate="inside",
                    object="drawer_1",
                    confidence=1.0,
                )
            ],
        ),
    )
    assert any(
        r.subject == "apple_1" and r.predicate == "inside" and r.object == "drawer_1"
        for r in added.relations
    )
    removed = manager.apply_event(
        added,
        WorldEvent(
            event_type="removed",
            description="remove on",
            relation_changes=[
                RelationChange(
                    action="remove",
                    subject="apple_1",
                    predicate="on",
                    object="table_1",
                    confidence=1.0,
                )
            ],
        ),
    )
    assert not any(
        r.subject == "apple_1" and r.predicate == "on" and r.object == "table_1"
        for r in removed.relations
    )


def test_low_confidence_does_not_overwrite_high_confidence():
    world = build_demo_world("demo")
    world.objects["drawer_1"].confidence = 0.95
    world.objects["drawer_1"].attributes["state"] = "closed"
    event = WorldEvent(
        event_type="noisy",
        description="low conf open",
        attribute_changes=[
            AttributeChange(
                entity_id="drawer_1",
                attribute="state",
                old_value="closed",
                new_value="open",
                confidence=0.2,
            )
        ],
    )
    updated = WorldModelManager().apply_event(world, event)
    assert updated.objects["drawer_1"].attributes["state"] == "closed"
    assert PROTECT_THRESHOLD <= 0.95


def test_new_object_inserted():
    world = build_demo_world("demo")
    event = WorldEvent(
        event_type="seen",
        description="cup appeared",
        new_objects=[ObjectState(id="cup_1", type="cup", confidence=0.9)],
    )
    updated = WorldModelManager().apply_event(world, event)
    assert "cup_1" in updated.objects
    assert updated.objects["cup_1"].type == "cup"
