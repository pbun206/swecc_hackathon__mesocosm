"""Telephone game benchmark — information survives 9 rewrites?"""

from __future__ import annotations

from typing import Any

from bench_common.env_sdk.base import BaseEnv, StepResult

ORIGINAL = (
    "Three bear cubs lived with their Mama, but one kept wandering off and "
    "getting into the jam. Papa Frog tried to count his children, but there "
    "were too many and he always lost track. Grandma Elephant thought she was "
    "the best counter in the forest, but she always got distracted and forgot "
    "where she started. At dinner, all the parents tried to count the bowls, "
    "but the kids kept moving around. Nobody could agree on the right number. "
    "So they all just ate — except little cub Three, who somehow ended up "
    "with four bowls."
)

MAX_STEPS = 10


class MyEnv(BaseEnv):
    def __init__(self) -> None:
        self._step: int = 0
        self._last_summary: str = ""

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        self._step = 0
        self._last_summary = ORIGINAL
        return {
            "text": ORIGINAL,
            "instruction": "Paraphrase the above in a way anyone could understand. Retain all information as possible but make it as if you were telling another person, your entire output will be seen by high schoolers so do not mention things like 'paraphrase' or 'summarize' or 'information' or 'repeat'. Instead, paraphrase directly.",
        }

    def step(self, action: Any) -> StepResult:
        self._step += 1
        response = str(action).strip()

        if self._step < MAX_STEPS - 1:
            self._last_summary = response
            return StepResult(
                observation={
                    "text": response,
                    "instruction": "Paraphrase the above in a way anyone could understand. Retain all information as possible but make it as if you were telling another person, your entire output will be seen by high schoolers so do not mention things like 'paraphrase' or 'summarize' or 'information' or 'repeat'. Instead, paraphrase directly.",
                },
                reward=0.0,
                terminated=False,
                truncated=False,
                info={"step": str(self._step), "summary": response},
            )

        if self._step == MAX_STEPS - 1:
            self._last_summary = response
            return StepResult(
                observation={
                    "question": "How many cubs are there in the story? Note: although the input might be unclear, this is no trick quesion and there is an expected answer. If you still think it's impossible, just explain why",
                },
                reward=0.0,
                terminated=False,
                truncated=False,
                info={"step": str(self._step), "summary": response},
            )

        correct = "3" in response or "three" in response.lower()
        return StepResult(
            observation={"result": "done"},
            reward=1.0 if correct else 0.0,
            terminated=True,
            truncated=False,
            info={
                "step": str(self._step),
                "correct": str(correct),
                "final_answer": response,
                "last_summary_before_question": self._last_summary,
            },
        )
