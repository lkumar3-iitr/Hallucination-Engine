import pytest

from scenario_generator.backends.carla_backend.carla_adapter_v2 import (
    resolve_carla_asset,
)


@pytest.mark.parametrize(
    ("asset_key", "blueprint"),
    [
        ("vehicle.passenger_01", "vehicle.tesla.model3"),
        ("vehicle.passenger_02", "vehicle.nissan.patrol_2021"),
        ("vehicle.bus_01", "vehicle.mitsubishi.fusorosa"),
        ("pedestrian.person_01", "walker.pedestrian.0001"),
    ],
)
def test_canonical_asset_key_resolves_to_carla_blueprint(asset_key, blueprint):
    assert resolve_carla_asset(asset_key) == blueprint


def test_explicit_carla_blueprint_remains_supported():
    assert resolve_carla_asset("vehicle.audi.tt") == "vehicle.audi.tt"
