from HallucinationEngine.scenarios.wrong_way_vehicle import WrongWayVehicleScenario
from HallucinationEngine.scenarios.stopped_vehicle import StoppedVehicleScenario


class ScenarioManager:
    """
    Manages hallucination scenarios.

    Responsibilities:
        1. Create the selected scenario.
        2. Update the active scenario every frame.
        3. Return hallucination metadata.
    """

    def __init__(self, config):
        self.config = config
        self.scenario = self._create_scenario(config.scenario_type)

    def _create_scenario(self, scenario_type):
        """
        Create scenario object based on scenario type.
        """

        if scenario_type == "wrong_way_vehicle":
            return WrongWayVehicleScenario(self.config)

        if scenario_type == "stopped_vehicle":
            return StoppedVehicleScenario(self.config)

        raise ValueError(f"Unknown scenario type: {scenario_type}")

    def update(self, metadata):
        if self.scenario is None:
            return None

        return self.scenario.update(metadata)

    def reset(self):
        self.scenario = self._create_scenario(self.config.scenario_type)