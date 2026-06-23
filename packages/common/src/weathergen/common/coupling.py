import dataclasses

@dataclasses.dataclass
class ModelCheckpoint:
    _checkpoint_str: dataclasses.InitVar[str]
    run_id: str = dataclasses.field(init=False)
    mini_epoch: int = dataclasses.field(init=False)
    model_timestep: int #!?
    
    def __post_init__(self, _checkpoint_str: str):
        run_id, mini_epoch = _checkpoint_str.split("@")
        self.run_id = run_id
        self.mini_epoch = int(mini_epoch)

@dataclasses.dataclass
class Coupling:
    name: str
    producer: str
    consumer: str
    stream: str

@dataclasses.dataclass
class Couplings:
    components: dict[str, ModelCheckpoint]
    couplings: dict[str, Coupling]
    
    @classmethod
    def from_args(cls, components: list[str]):
        components_kv = (component.split("=") for component in components)
        components = {
            component: ModelCheckpoint(checkpoint_str) for component, checkpoint_str in components_kv
        }
        
        return cls(components)
    
    def generate_run_configs():