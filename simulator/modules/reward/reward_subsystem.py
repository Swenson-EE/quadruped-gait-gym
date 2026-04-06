from simulator.core.subsystem.subsystem import Subsystem

import numpy as np


class RewardSubsystem(Subsystem):

    

    def __init__(self, parent):
        super().__init__(parent)

        self._normalization_factor = {
            "reward": {},
            "penalty": {}
        }
        self._weight = {
            "reward": {},
            "penalty": {}
        }

        self._reducers = {
            "reward": {},
            "penalty": {}
        }



    def _get_components(self):
        return None

    

    
    def get(self):
        #reward, penalty = (self._get_components() / self._normalization_factor) * self._weight
        
        components = self._get_components()
        # reward, penalty = (
        #     {
        #         k: (v / norm.get(k, 1.0)) * weight.get(k, 1.0)
        #         for k, v in (comp or {}).items()
        #     } if comp is not None else None
        #     for comp, norm, weight in zip(
        #         components,
        #         (self._normalization_factor["reward"], self._normalization_factor["penalty"]),
        #         (self._weight["reward"], self._weight["penalty"])
        #     )
        # )

        reward, penalty = (
            {
                k: (
                    reducers.get(k, lambda x: x)(v / norm.get(k, 1.0))
                ) * weight.get(k, 1.0)
                for k, v in (comp or {}).items()
            } if comp is not None else None
            for comp, norm, weight, reducers in zip(
                components,
                (self._normalization_factor["reward"], self._normalization_factor["penalty"]),
                (self._weight["reward"], self._weight["penalty"]),
                (self._reducers["reward"], self._reducers["penalty"])
            )
        )

        return reward, penalty

    def get_normalized(self):
        components = self._get_components()

        reward, penalty = (
            {
                k: (
                    v / norm.get(k, 1.0) 
                ) for k, v in (comp or {}).items()
            } if comp is not None else None
            for comp, norm in zip(
                components,
                (self._normalization_factor['reward'], self._normalization_factor['penalty'])
            )
        )

        return reward, penalty