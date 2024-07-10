from abc import ABC, abstractmethod

class EvalModel(ABC):
    @abstractmethod
    def is_ntg(self):
        return None

    @abstractmethod
    def eval_with_cross_validation(self, dataset, opt: dict, k=5):
        pass