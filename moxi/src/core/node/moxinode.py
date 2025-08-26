from moxi.src.core.interface import AbstractMoxiNode
from moxi.src.trainers import PytorchTrainer, ScikitLearnTrainer, DummyTrainer
from moxi.src.trainers.util import convert_str_to_framework

class MoxiClientNode(AbstractMoxiNode):
    def __init__(self, node_id, node_type, parent_network):
        super().__init__(node_id, node_type,parent_network)
        # Additional initialization for client nodes can be added here

    def initialize(self,  model, data, config):
        # select trainer based on config
        if config["framework"] == "pytorch":
            trainer = PytorchTrainer
            config["framework"] = convert_str_to_framework(config["framework"])

        elif config["framework"] == "sklearn":
            trainer = ScikitLearnTrainer
            config["framework"] = convert_str_to_framework(config["framework"])

        else:
            trainer = DummyTrainer

        self.trainer = trainer(self.node_id, model, data, config)

    
    def receive_model_update(self, update):
        return super().receive_model_update(update)
    
    def send_model_update(self, target_node_id, update):
        return super().send_model_update(target_node_id, update)
    
    def train_local_model(self):
        # check if trainer is initialized
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call initialize() first.")

        # local training 
        self.trainer.train()


        # update distributed metadata
        self.current_round += 1
        self.model_version += 1
    
    def set_data(self, data):
        self.trainer.data = data
        return True
    
    def set_model(self, model):
        self.trainer.model = model
        return True