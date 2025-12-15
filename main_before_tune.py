import asyncio
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

MAX_TOKENS = 500

BUGGY_ADVANTAGE_CODE = """from __future__ import annotations


def compute_gae(rewards, values, terminated, truncated, gamma, lam):
    T = len(rewards)
    if len(values) != T + 1:
        raise ValueError("values must have length T + 1")
    if T == 0:
        return []
    B = len(rewards[0])

    advantages = [[0.0 for _ in range(B)] for _ in range(T)]
    next_adv = [0.0 for _ in range(B)]

    for t in range(T - 1, -1, -1):
        for b in range(B):
            done = bool(terminated[t][b]) or bool(truncated[t][b])
            mask = 0.0 if done else 1.0
            delta = rewards[t][b] + gamma * values[t + 1][b] * mask - values[t][b]
            boundary = 0.0 if bool(terminated[t][b]) else 1.0
            advantages[t][b] = delta + gamma * lam * next_adv[b] * boundary
        next_adv = advantages[t]

    return advantages
"""


class ReadFileToolResult(TypedDict):
    content: str | None
    error: str | None


class WriteFileToolResult(TypedDict):
    ok: bool
    error: str | None


class RunChecksToolResult(TypedDict):
    public_checks_passed: bool
    hidden_checks_passed: bool
    failure_messages: list[str]


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def _reset_advantage_file() -> None:
    Path("advantage.py").write_text(BUGGY_ADVANTAGE_CODE, encoding="utf8")


def read_file_tool(path: str) -> ReadFileToolResult:
    if path != "advantage.py":
        return {"content": None, "error": "can only read advantage.py"}
    try:
        return {"content": Path(path).read_text(encoding="utf8"), "error": None}
    except Exception as e:
        return {"content": None, "error": str(e)}


def write_file_tool(path: str, content: str) -> WriteFileToolResult:
    if path != "advantage.py":
        return {"ok": False, "error": "can only write advantage.py"}
    try:
        Path(path).write_text(content, encoding="utf8")
        return {"ok": True, "error": None}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    return {"answer": answer, "submitted": True}


def _load_candidate_module() -> Any:
    p = Path("advantage.py").resolve()
    unique = f"advantage_candidate_{os.getpid()}_{int(time.time() * 1_000_000)}_{random.randint(0, 10**9)}"
    spec = importlib.util.spec_from_file_location(unique, p)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load advantage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compute_gae_ref(rewards, values, terminated, truncated, gamma, lam):
    T = len(rewards)
    if len(values) != T + 1:
        raise ValueError("values must have length T + 1")
    if T == 0:
        return []

    B = len(rewards[0])
    for t in range(T):
        if len(rewards[t]) != B or len(terminated[t]) != B or len(truncated[t]) != B:
            raise ValueError("shape mismatch")
    for t in range(T + 1):
        if len(values[t]) != B:
            raise ValueError("shape mismatch")

    gamma_is_seq = isinstance(gamma, list)
    lam_is_seq = isinstance(lam, list)
    if gamma_is_seq and len(gamma) != T:
        raise ValueError("gamma list must have length T")
    if lam_is_seq and len(lam) != T:
        raise ValueError("lam list must have length T")

    adv = [[0.0 for _ in range(B)] for _ in range(T)]
    next_adv = [0.0 for _ in range(B)]

    for t in range(T - 1, -1, -1):
        g = float(gamma[t]) if gamma_is_seq else float(gamma)
        l = float(lam[t]) if lam_is_seq else float(lam)
        cur = [0.0 for _ in range(B)]
        for b in range(B):
            boot = 0.0 if bool(terminated[t][b]) else 1.0
            delta = rewards[t][b] + g * values[t + 1][b] * boot - values[t][b]
            cont = 0.0 if (bool(terminated[t][b]) or bool(truncated[t][b])) else 1.0
            cur[b] = delta + g * l * next_adv[b] * cont
        adv[t] = cur
        next_adv = cur
    return adv


def _max_abs_diff(a, b) -> float:
    m = 0.0
    for t in range(len(a)):
        for j in range(len(a[t])):
            d = abs(float(a[t][j]) - float(b[t][j]))
            if d > m:
                m = d
    return m


def _gen_case(seed: int, T: int, B: int):
    rng = random.Random(seed)
    rewards = [[rng.uniform(-1.0, 1.0) for _ in range(B)] for _ in range(T)]
    values = [[rng.uniform(-1.0, 1.0) for _ in range(B)] for _ in range(T + 1)]

    terminated = [[False for _ in range(B)] for _ in range(T)]
    truncated = [[False for _ in range(B)] for _ in range(T)]

    for t in range(T):
        for b in range(B):
            if rng.random() < 0.12:
                terminated[t][b] = True
            elif rng.random() < 0.12:
                truncated[t][b] = True

    return rewards, values, terminated, truncated


def _run_checks() -> RunChecksToolResult:
    failures_public: list[str] = []
    failures_hidden: list[str] = []

    try:
        mod = _load_candidate_module()
        fn = getattr(mod, "compute_gae", None)
        if fn is None or not callable(fn):
            return {
                "public_checks_passed": False,
                "hidden_checks_passed": False,
                "failure_messages": ["missing_compute_gae"],
            }
    except Exception as e:
        return {
            "public_checks_passed": False,
            "hidden_checks_passed": False,
            "failure_messages": [f"import_error:{type(e).__name__}"],
        }

    try:
        try:
            mod.compute_gae([[0.0]], [[0.0]], [[False]], [[False]], 0.9, 0.95)
            failures_public.append("values_length_should_raise")
        except ValueError:
            pass
        except Exception:
            failures_public.append("values_length_wrong_exception")

        try:
            mod.compute_gae([[0.0], [0.0, 0.0]], [[0.0], [0.0], [0.0]], [[False], [False, False]], [[False], [False, False]], 0.9, 0.95)
            failures_public.append("shape_should_raise")
        except ValueError:
            pass
        except Exception:
            failures_public.append("shape_wrong_exception")


        rewards = [[1.0], [1.0], [1.0]]
        values = [[0.0], [0.5], [0.0], [0.0]]
        terminated = [[False], [False], [True]]
        truncated = [[False], [True], [False]]
        gamma = 0.9
        lam = 0.95

        before = (repr(rewards), repr(values), repr(terminated), repr(truncated))
        _ = mod.compute_gae(rewards, values, terminated, truncated, gamma, lam)
        after = (repr(rewards), repr(values), repr(terminated), repr(truncated))
        if before != after:
            failures_public.append("mutated_inputs")
        expected = [[1.8775], [0.5], [1.0]]
        got = mod.compute_gae(rewards, values, terminated, truncated, gamma, lam)
        if not (isinstance(got, list) and len(got) == 3 and len(got[0]) == 1):
            failures_public.append("example_shape_wrong")
        else:
            if _max_abs_diff(got, expected) > 1e-6:
                failures_public.append("example_mismatch")

        rewards, values, terminated, truncated = _gen_case(123, 12, 4)
        expected = _compute_gae_ref(rewards, values, terminated, truncated, 0.99, 0.95)
        got = mod.compute_gae(rewards, values, terminated, truncated, 0.99, 0.95)
        gamma_list = [0.85 + 0.01 * (t % 5) for t in range(12)]
        lam_list = [0.7 + 0.02 * (t % 3) for t in range(12)]
        expected2 = _compute_gae_ref(rewards, values, terminated, truncated, gamma_list, lam_list)
        got2 = mod.compute_gae(rewards, values, terminated, truncated, gamma_list, lam_list)
        if _max_abs_diff(got2, expected2) > 1e-5:
            failures_public.append("gamma_lam_list_public_mismatch")

        if not (isinstance(got, list) and len(got) == 12 and len(got[0]) == 4):
            failures_public.append("random_public_shape_wrong")
        else:
            if _max_abs_diff(got, expected) > 1e-5:
                failures_public.append("random_public_mismatch")

        rewards = [[0.0], [0.0], [0.0], [0.0]]
        values = [[0.2], [0.4], [0.6], [0.8], [1.0]]
        terminated = [[False], [False], [False], [False]]
        truncated = [[False], [True], [False], [False]]
        expected = _compute_gae_ref(rewards, values, terminated, truncated, 0.9, 0.9)
        got = mod.compute_gae(rewards, values, terminated, truncated, 0.9, 0.9)
        if _max_abs_diff(got, expected) > 1e-6:
            failures_public.append("boundary_truncation_mismatch")

        rewards = [[0.3], [0.3], [0.3], [0.3]]
        values = [[0.1], [0.2], [0.3], [0.4], [0.5]]
        terminated = [[False], [True], [False], [False]]
        truncated = [[False], [False], [False], [False]]
        expected = _compute_gae_ref(rewards, values, terminated, truncated, 0.9, 0.9)
        got = mod.compute_gae(rewards, values, terminated, truncated, 0.9, 0.9)
        if _max_abs_diff(got, expected) > 1e-6:
            failures_public.append("boundary_termination_mismatch")

    except Exception as e:
        failures_public.append(f"public_exception:{type(e).__name__}")

    try:
        rewards, values, terminated, truncated = _gen_case(9991, 18, 3)
        expected = _compute_gae_ref(rewards, values, terminated, truncated, 0.97, 0.8)
        got = mod.compute_gae(rewards, values, terminated, truncated, 0.97, 0.8)
        gamma_list = [0.9 + 0.005 * (t % 4) for t in range(18)]
        lam_list = [0.6 + 0.03 * (t % 5) for t in range(18)]
        expected2 = _compute_gae_ref(rewards, values, terminated, truncated, gamma_list, lam_list)
        got2 = mod.compute_gae(rewards, values, terminated, truncated, gamma_list, lam_list)
        if _max_abs_diff(got2, expected2) > 1e-5:
            failures_hidden.append("gamma_lam_list_hidden_mismatch")

        if _max_abs_diff(got, expected) > 1e-5:
            failures_hidden.append("random_hidden_mismatch")

        rewards = [[0.0], [0.0], [0.0]]
        values = [[10.0], [9.0], [8.0], [7.0]]
        terminated = [[False], [True], [False]]
        truncated = [[False], [True], [False]]
        expected = _compute_gae_ref(rewards, values, terminated, truncated, 0.99, 0.95)
        got = mod.compute_gae(rewards, values, terminated, truncated, 0.99, 0.95)
        if _max_abs_diff(got, expected) > 1e-6:
            failures_hidden.append("both_true_boundary_mismatch")

    except Exception as e:
        failures_hidden.append(f"hidden_exception:{type(e).__name__}")

    return {
        "public_checks_passed": len(failures_public) == 0,
        "hidden_checks_passed": len(failures_hidden) == 0,
        "failure_messages": failures_public + failures_hidden,
    }


def run_checks_tool() -> RunChecksToolResult:
    return _run_checks()


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Any],
    max_steps: int = 20,
    model: str | None = None,
    verbose: bool = True,
) -> Any | None:
    if model is None:
        model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
        )

        assert response.stop_reason in ["max_tokens", "tool_use", "end_turn"], (
            f"unsupported stop_reason {response.stop_reason}"
        )

        has_tool_use = False
        tool_results = []
        submitted_answer = None

        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(content.text)
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name
                tool_input = content.input
                handler = tool_handlers.get(tool_name)

                if handler is None:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps({"error": f"unknown tool {tool_name}"}),
                        }
                    )
                    continue

                if tool_name == "submit_answer":
                    if not (isinstance(tool_input, dict) and "answer" in tool_input):
                        result = {"error": "submit_answer requires answer"}
                    else:
                        result = handler(tool_input["answer"])
                        submitted_answer = result.get("answer")
                else:
                    try:
                        if isinstance(tool_input, dict):
                            result = handler(**tool_input)
                        else:
                            result = handler(tool_input)
                    except Exception as e:
                        result = {"error": f"tool_error:{type(e).__name__}:{e}"}

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": json.dumps(result),
                    }
                )

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            if submitted_answer is not None:
                return submitted_answer
        else:
            break

    return None


async def run_single_test(run_id: int, num_runs: int, tools, tool_handlers, verbose: bool = False):
    _reset_advantage_file()

    prompt = """You are debugging an RL training bug.

Edit advantage.py so compute_gae matches this contract:

Inputs:
- rewards: T x B list of floats
- values: (T + 1) x B list of floats
- terminated: T x B list of bool
- truncated: T x B list of bool
- gamma: float OR list of length T
- lam: float OR list of length T

Output:
- advantages: T x B list of floats

Definitions:
- For each t,b:
  g = gamma[t] if gamma is a list else gamma
  l = lam[t] if lam is a list else lam
  boot = 0 if terminated[t][b] else 1
  delta[t,b] = rewards[t][b] + g * values[t+1][b] * boot - values[t][b]
- Let done[t,b] = terminated[t][b] or truncated[t][b]
  advantages recursion only continues when done is false:
  adv[t,b] = delta[t,b] + g * l * adv[t+1,b] if done[t,b] is false
  adv[t,b] = delta[t,b] if done[t,b] is true
- If both terminated and truncated are true at a timestep, treat it as terminated for bootstrapping and as a boundary for recursion.

Requirements:
- Must raise ValueError if len(values) != T + 1
- Must raise ValueError if any row in rewards, terminated, truncated has length different from B
- Must raise ValueError if any row in values has length different from B
- If gamma is a list, must raise ValueError unless len(gamma) == T
- If lam is a list, must raise ValueError unless len(lam) == T
- Must not mutate rewards, values, terminated, or truncated
- Must work for B = 1 and general B

Use tools:
- read_file (advantage.py)
- write_file (advantage.py)
- run_checks
When run_checks passes, call submit_answer with any value.
"""

    started = time.time()
    await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=4,
        verbose=verbose,
    )
    duration = time.time() - started

    report = _run_checks()
    success = report["public_checks_passed"] and report["hidden_checks_passed"]

    if success:
        print(f"✓ Run {run_id}: SUCCESS ({duration:.1f}s)")
    else:
        msg = ", ".join(report["failure_messages"][:3])
        if len(report["failure_messages"]) > 3:
            msg += ", ..."
        print(f"✗ Run {run_id}: FAILURE ({duration:.1f}s) - {msg}")

    return run_id, success, report


async def main():
    tools: list[ToolUnionParam] = [
        {
            "name": "read_file",
            "description": "Read advantage.py",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write advantage.py",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
        {
            "name": "run_checks",
            "description": "Run checks. Returns booleans and short failure codes.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "submit_answer",
            "description": "Submit final answer",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"description": "Any value"}},
                "required": ["answer"],
            },
        },
    ]

    print(f"Using model: {os.getenv('ANTHROPIC_MODEL', 'claude-haiku-4-5')}")

    tool_handlers = {
        "read_file": read_file_tool,
        "write_file": write_file_tool,
        "run_checks": run_checks_tool,
        "submit_answer": submit_answer_tool,
    }

    if "--check" in sys.argv:
        _reset_advantage_file()
        report = _run_checks()
        print(json.dumps(report, indent=2))
        return

    num_runs = int(os.getenv("NUM_RUNS", "10"))
    print(f"Running {num_runs} test iterations sequentially...")
    print("=" * 60)

    results = []
    for i in range(num_runs):
        result = await run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            tools=tools,
            tool_handlers=tool_handlers,
            verbose=False,
        )
        results.append(result)

    successes = sum(success for _, success, _ in results)
    pass_rate = (successes / num_runs) * 100

    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
