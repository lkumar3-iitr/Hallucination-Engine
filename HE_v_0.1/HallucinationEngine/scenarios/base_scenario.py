from abc import ABC, abstractmethod


class BaseScenario(ABC):
    """
    Base class for all hallucination scenarios.

    Every scenario should:
    1. Decide whether it is active.
    2. Maintain temporal consistency.
    3. Return hallucinated object metadata.
    """

    def __init__(self, config):
        self.config = config
        self.active = False
        self.start_frame = None

    @abstractmethod
    def update(self, metadata):
        """
        Update the scenario for the current frame.

        Args:
            metadata: dictionary containing frame id, ego state, route info, etc.

        Returns:
            hallucination dictionary or None
        """
        pass

    def should_start(self, frame_id):
        return frame_id >= self.config.trigger_frame

    def is_within_duration(self, frame_id):
        if self.start_frame is None:
            return False

        elapsed = frame_id - self.start_frame
        return elapsed <= self.config.duration_frames