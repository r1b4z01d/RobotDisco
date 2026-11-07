from __future__ import annotations


class Backend:
    """observe -> act, once per control step. `act` returns one 10-D action (the backend keeps any chunk)."""
    name = "base"

    def load(self, checkpoint: str | None) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def act(self, state, images: dict, task: str):
        raise NotImplementedError

    def info(self) -> dict:
        return {"backend": self.name}
