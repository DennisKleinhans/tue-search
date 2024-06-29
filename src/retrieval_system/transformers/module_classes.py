from abc import ABCMeta, abstractmethod

class ProcessingModule(metaclass=ABCMeta):
    def __init__(self, train_config, model_config) -> None:
        self.train_config = train_config
        self.model_config = model_config
        
    @abstractmethod
    def execute(self):
        pass