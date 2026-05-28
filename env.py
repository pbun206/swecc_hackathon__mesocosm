"""Minesweeper benchmark — can an LLM safely clear a minefield?"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from bench_common.env_sdk.base import BaseEnv, StepResult

ROWS = 5
COLS = 5
NUM_MINES = 5
SAFE_CELLS = ROWS * COLS - NUM_MINES
MAX_STEPS = 30
SYSTEM = (
    "You are playing Minesweeper. Output ONLY a JSON object on a single line: "
    '{"row": r, "col": c}. No explanation, no commentary, no extra text.'
)


class MyEnv(BaseEnv):
    def __init__(self) -> None:
        self._board: list[list[bool]] = []
        self._revealed: list[list[bool]] = []
        self._step: int = 0
        self._safe_revealed: int = 0
        self._mines_placed: bool = False
        self._rng: random.Random = random.Random()

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        self._rng = random.Random(seed)
        self._step = 0
        self._safe_revealed = 0
        self._mines_placed = False

        self._board = [[False] * COLS for _ in range(ROWS)]
        self._revealed = [[False] * COLS for _ in range(ROWS)]

        return {
            "board": self._render_board(),
            "instruction": (
                f"Minesweeper. {ROWS}x{COLS} grid, {NUM_MINES} mines. "
                f"Numbers show adjacent mine count. '?' = hidden. "
                f"Do NOT click on a number — only click on '?' cells. "
                f"Reveal all {SAFE_CELLS} safe cells to win. "
                f"Your first move is always safe. "
                f"Respond with ONLY JSON, no other text. "
                f'Example: {{\"row\": 2, \"col\": 3}}'
            ),
        }

    def _place_mines(self, safe_row: int, safe_col: int) -> None:
        excluded = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = safe_row + dr, safe_col + dc
                if 0 <= r < ROWS and 0 <= c < COLS:
                    excluded.add((r, c))

        candidates = [
            (r, c) for r in range(ROWS) for c in range(COLS)
            if (r, c) not in excluded
        ]
        for r, c in self._rng.sample(candidates, min(NUM_MINES, len(candidates))):
            self._board[r][c] = True
        self._mines_placed = True

    def step(self, action: Any) -> StepResult:
        self._step += 1

        row, col, err = self._parse_action(action)
        if err:
            return StepResult(
                observation={
                    "board": self._render_board(),
                    "result": f"Illegal move — {err} Game over.",
                },
                reward=0.0,
                terminated=True,
                truncated=False,
                info={"step": str(self._step), "result": "illegal_move"},
            )

        if self._revealed[row][col]:
            return StepResult(
                observation={
                    "board": self._render_board(),
                    "result": "Illegal move — cell already revealed. Game over.",
                },
                reward=0.0,
                terminated=True,
                truncated=False,
                info={"step": str(self._step), "result": "illegal_move"},
            )

        if not self._mines_placed:
            self._place_mines(row, col)

        if self._board[row][col]:
            self._revealed[row][col] = True
            return StepResult(
                observation={
                    "board": self._render_board(),
                    "result": "BOOM! You hit a mine.",
                },
                reward=0.0,
                terminated=True,
                truncated=False,
                info={
                    "step": str(self._step),
                    "result": "mine_hit",
                    "cleared": str(self._safe_revealed),
                    "total_safe": str(SAFE_CELLS),
                },
            )

        newly = self._flood_reveal(row, col)
        step_reward = newly / SAFE_CELLS
        won = self._safe_revealed == SAFE_CELLS

        return StepResult(
            observation={
                "board": self._render_board(),
                "cells_revealed": newly,
                "safe_remaining": SAFE_CELLS - self._safe_revealed,
                "reminder": "Only click '?' cells. Do NOT click on a number.",
            },
            reward=step_reward,
            terminated=won,
            system_prompt=None if won else SYSTEM,
            truncated=not won and self._step >= MAX_STEPS,
            info={
                "step": str(self._step),
                "cleared": str(self._safe_revealed),
                "total_safe": str(SAFE_CELLS),
                "won": str(won),
            },
        )

    def _parse_action(self, action: Any) -> tuple[int, int, str | None]:
        try:
            if isinstance(action, str):
                match = re.search(r"\{[^}]+\}", action)
                if match:
                    action = json.loads(match.group())
                else:
                    action = json.loads(action)
            row, col = int(action["row"]), int(action["col"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return 0, 0, 'Invalid action. Send JSON: {"row": r, "col": c}'

        if not (0 <= row < ROWS and 0 <= col < COLS):
            return 0, 0, f"Out of bounds. row: 0-{ROWS - 1}, col: 0-{COLS - 1}"

        return row, col, None

    def _adjacent_mines(self, row: int, col: int) -> int:
        count = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < ROWS and 0 <= c < COLS and self._board[r][c]:
                    count += 1
        return count

    def _flood_reveal(self, row: int, col: int) -> int:
        if self._revealed[row][col] or self._board[row][col]:
            return 0

        stack = [(row, col)]
        count = 0
        while stack:
            r, c = stack.pop()
            if self._revealed[r][c]:
                continue
            self._revealed[r][c] = True
            count += 1
            self._safe_revealed += 1

            if self._adjacent_mines(r, c) == 0:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < ROWS and 0 <= nc < COLS and not self._revealed[nr][nc]:
                            stack.append((nr, nc))
        return count

    def _render_board(self) -> str:
        lines = ["  " + " ".join(str(c) for c in range(COLS))]
        for r in range(ROWS):
            row_str = f"{r} "
            for c in range(COLS):
                if self._revealed[r][c]:
                    if self._board[r][c]:
                        row_str += "* "
                    else:
                        row_str += f"{self._adjacent_mines(r, c)} "
                else:
                    row_str += "? "
            lines.append(row_str.rstrip())
        return "\n".join(lines)
