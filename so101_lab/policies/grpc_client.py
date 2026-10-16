# Runs in: isaaclab-env (Python 3.11)
"""Synchronous gRPC client for remote policy inference."""

import os
import pickle  # nosec
import sys
from pathlib import Path
from typing import Iterator

import io

import numpy as np


# 本地协议用 grpcio-tools==1.62.3 生成，兼容 Isaac 的 Protobuf 4.25。
# 推理客户端无需导入 Python 3.12 环境中的 LeRobot 包。
import grpc
from so101_lab.transport import services_pb2, services_pb2_grpc
from so101_lab.transport.utils import send_bytes_in_chunks, grpc_channel_options


class GrpcPolicyClient:
    def __init__(
        self,
        checkpoint: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        policy_type: str = "smolvla",
        task: str = "Place the cube into the matching slot on the platform",
        actions_per_chunk: int = 15,
        device: str = "cuda",
        noise_prior: str | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.host = host
        self.port = port
        self.policy_type = policy_type
        self.task = task
        self.actions_per_chunk = actions_per_chunk
        self.device = device
        self.noise_prior = noise_prior

        self._channel = None
        self._stub = None
        self._action_chunk: np.ndarray | None = None
        self._chunk_idx: int = 0
        self._timestep: int = 0

    def connect(self) -> None:
        options = grpc_channel_options()
        self._channel = grpc.insecure_channel(f"{self.host}:{self.port}", options=options)
        self._stub = services_pb2_grpc.AsyncInferenceStub(self._channel)

        self._stub.Ready(services_pb2.Empty())

        specs = pickle.dumps({  # nosec
            "checkpoint": self.checkpoint,
            "policy_type": self.policy_type,
            "task": self.task,
            "actions_per_chunk": self.actions_per_chunk,
            "device": self.device,
            "noise_prior": self.noise_prior,
        })
        self._stub.SendPolicyInstructions(services_pb2.PolicySetup(data=specs))

    def reset(self) -> None:
        self._action_chunk = None
        self._chunk_idx = 0
        self._timestep = 0

    def select_action(self, obs: dict) -> np.ndarray:
        if self._action_chunk is None or self._chunk_idx >= self.actions_per_chunk:
            obs_with_meta = dict(obs)
            obs_with_meta["timestep"] = self._timestep
            obs_bytes = pickle.dumps(obs_with_meta)  # nosec

            self._stub.SendObservations(
                send_bytes_in_chunks(obs_bytes, services_pb2.Observation)
            )

            response = self._stub.GetActions(services_pb2.Empty())
            self._action_chunk = np.load(io.BytesIO(response.data))
            self._chunk_idx = 0

        action = self._action_chunk[self._chunk_idx]
        self._chunk_idx += 1
        self._timestep += 1
        return action

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None
