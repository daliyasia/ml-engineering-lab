import os
from lightning.pytorch.plugins.environments import ClusterEnvironment

class MyClusterEnvironment(ClusterEnvironment):
    '''def __init__(self):
        super().__init__()  '''
    
    def creates_processes_externally(self) -> bool:
        pass
    def detect() -> bool:
        pass
    def main_address(self) -> str:
        return os.environ["MASTER_ADDR"]
    def main_port(self) -> int:
        return int(os.environ["MASTER_PORT"])
    def set_global_rank(self, rank: int) -> None:
        pass
    def set_world_size(self, size: int) -> None:
        pass
    @property
    def creates_children(self) -> bool:
            # return True if the cluster is managed 
            # (you don't launch processes yourself)
            return True
    def world_size(self) -> int:
            #print(os.environ["WORLD_SIZE"])
            return int(os.environ["WORLD_SIZE"])
    def global_rank(self) -> int:
            #print(os.environ["RANK"])
            return int(os.environ["RANK"])
    def local_rank(self) -> int:
            #print(os.environ["LOCAL_RANK"])
            return int(os.environ["LOCAL_RANK"])
    def node_rank(self) -> int:
            #print(os.environ["NODE_RANK"])
            return int(os.environ["NODE_RANK"])
    def master_address(self) -> str:
            #print("master addr", os.environ["MASTER_ADDR"])
            return os.environ["MASTER_ADDR"]
    def master_port(self) -> int:
            #print("master port", os.environ["MASTER_PORT"])
            return int(os.environ["MASTER_PORT"])