# agents/base_agent.py

from abc import ABC, abstractmethod
class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def remember(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def replay(self):
        pass
