import ast
import asyncio
import hashlib
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

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "600"))
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "0"))
random.seed(GLOBAL_SEED)

COMPUTE_GAE_SPEC = """
Function: compute_gae(rewards, values, terminated, truncated, gamma, lam) -> advantages

Inputs
- rewards: T x B list of lists (float)
- values: (T+1) x B list of lists (float) - includes bootstrap value at index T
- terminated: T x B list of lists (bool) - true episode termination
- truncated: T x B list of lists (bool) - time-limit truncation (not true termination)
- gamma: discount factor (float)
- lam: GAE lambda parameter (float)

Output
- advantages: T x B list of lists (float)

Validation
Raise ValueError if len(values) != len(rewards) + 1.

Behavior
Implement the Generalized Advantage Estimation algorithm. Process timesteps backward.
The key distinction is how terminated vs truncated affect the TD error and advantage propagation:
- terminated=True: The episode truly ended. Do not use the next value for bootstrapping.
- truncated=True: Time limit was hit but episode didn't truly end. The next value IS valid for bootstrapping.
- Both flags affect whether to propagate the accumulated advantage to earlier timesteps.

"""
BUGGY_ADVANTAGE_CODE = """
from __future__ import annotations


def compute_gae(
    rewards: list[list[float]],
    values: list[list[float]],
    terminated: list[list[bool]],
    truncated: list[list[bool]],
    gamma: float | list[float],
    lam: float | list[float],
) -> list[list[float]]:
    T = len(rewards)
    if len(values) != T + 1:
        raise ValueError("values must have length T+1")
    if T == 0:
        return []

    def _as_schedule(x: float | list[float], name: str) -> list[float]:
        if isinstance(x, (int, float)):
            return [float(x)] * T
        if isinstance(x, list):
            if len(x) != T:
                raise ValueError(f"{name} must have length T")
            return [float(v) for v in x]
        raise TypeError(f"{name} must be float or list[float]")

    gammas = _as_schedule(gamma, "gamma")
    lams = _as_schedule(lam, "lam")

    B = len(rewards[0])
    advantages = [[0.0] * B for _ in range(T)]
    next_adv = [0.0] * B

    for t in range(T - 1, -1, -1):
        g = gammas[t]
        l = lams[t]
        for b in range(B):
            term = bool(terminated[t][b])
            trunc = bool(truncated[t][b])
            delta = rewards[t][b] + g * values[t + 1][b] * (0.0 if (term or trunc) else 1.0) - values[t][b]
            continue_mask = 0.0 if term else (0.5 if trunc else 1.0)
            next_adv[b] = delta + g * l * continue_mask * next_adv[b]
            advantages[t][b] = next_adv[b]

    return [row[:] for row in advantages]

"""

REFERENCE_ADVANTAGE_CODE = """
from __future__ import annotations


def compute_gae(
    rewards: list[list[float]],
    values: list[list[float]],
    terminated: list[list[bool]],
    truncated: list[list[bool]],
    gamma: float,
    lam: float,
) -> list[list[float]]:
    T = len(rewards)
    if len(values) != T + 1:
        raise ValueError("values must have length T+1")
    if T == 0:
        return []

    B = len(rewards[0])
    adv = [[0.0 for _ in range(B)] for _ in range(T)]
    next_adv = [0.0 for _ in range(B)]

    for t in range(T - 1, -1, -1):
        cur = [0.0 for _ in range(B)]
        for b in range(B):
            boot = 0.0 if bool(terminated[t][b]) else 1.0
            delta = float(rewards[t][b]) + gamma * float(values[t + 1][b]) * boot - float(values[t][b])
            cont = 0.0 if (bool(terminated[t][b]) or bool(truncated[t][b])) else 1.0
            cur[b] = delta + gamma * lam * next_adv[b] * cont
        adv[t] = cur
        next_adv = cur
    return adv

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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    data = path.read_bytes()
    h.update(data)
    return h.hexdigest()


def read_file_tool() -> ReadFileToolResult:
    try:
        return {"content": Path("advantage.py").read_text(encoding="utf8"), "error": None}
    except Exception as e:
        return {"content": None, "error": str(e)}


def write_file_tool(content: str, path: str | None = None, **_: Any) -> WriteFileToolResult:
    try:
        target = Path(path) if path else Path("advantage.py")
        if target.name != "advantage.py":
            target = Path("advantage.py")
        target.write_text(content, encoding="utf8")
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
        raise ValueError("values must have length T+1")
    if T == 0:
        return []

    B = len(rewards[0])
    adv = [[0.0 for _ in range(B)] for _ in range(T)]
    next_adv = [0.0 for _ in range(B)]

    for t in range(T - 1, -1, -1):
        cur = [0.0 for _ in range(B)]
        for b in range(B):
            boot = 0.0 if bool(terminated[t][b]) else 1.0
            delta = float(rewards[t][b]) + gamma * float(values[t + 1][b]) * boot - float(values[t][b])
            cont = 0.0 if (bool(terminated[t][b]) or bool(truncated[t][b])) else 1.0
            cur[b] = delta + gamma * lam * next_adv[b] * cont
        adv[t] = cur
        next_adv = cur
    return adv


EPS = 1e-6


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
    except Exception as e:
        return {
            "public_checks_passed": False,
            "hidden_checks_passed": False,
            "failure_messages": [f"load_failed:{type(e).__name__}"],
        }

    fn = getattr(mod, "compute_gae", None)
    if not callable(fn):
        return {
            "public_checks_passed": False,
            "hidden_checks_passed": False,
            "failure_messages": ["missing_compute_gae"],
        }

    def _safe_max_abs_diff(a: Any, b: Any) -> float:
        try:
            if not isinstance(a, list) or not isinstance(b, list):
                return 1e9
            if len(a) != len(b):
                return 1e9
            worst = 0.0
            for row_a, row_b in zip(a, b):
                if not isinstance(row_a, list) or not isinstance(row_b, list):
                    return 1e9
                if len(row_a) != len(row_b):
                    return 1e9
                for x, y in zip(row_a, row_b):
                    worst = max(worst, abs(float(x) - float(y)))
            return worst
        except Exception:
            return 1e9

    def _expect_match(
        rewards: list[list[float]],
        values: list[list[float]],
        terminated: list[list[bool]],
        truncated: list[list[bool]],
        gamma: float,
        lam: float,
        mismatch_code: str,
        failures: list[str],
        exception_prefix: str,
    ) -> str | None:
        try:
            got = fn(rewards, values, terminated, truncated, gamma, lam)
        except Exception as e:
            failures.append(f"{exception_prefix}:{type(e).__name__}")
            return type(e).__name__
        expected = _compute_gae_ref(rewards, values, terminated, truncated, gamma, lam)
        if _safe_max_abs_diff(got, expected) > EPS:
            failures.append(mismatch_code)
        return None

    try:
        fn([[0.0]], [[0.0]], [[False]], [[False]], 0.9, 0.95)
        failures_public.append("values_len_should_raise")
    except ValueError:
        pass
    except Exception:
        failures_public.append("values_len_should_raise")

    _expect_match(
        [[1.0], [1.0], [1.0]],
        [[0.0], [0.5], [0.0], [0.0]],
        [[False], [False], [True]],
        [[False], [True], [False]],
        0.9, 0.95, "example_mismatch", failures_public, "public_exception",
    )

    _expect_match(
        [[0.0], [0.0], [0.0]],
        [[0.0], [1.0], [2.0], [3.0]],
        [[False], [False], [False]],
        [[False], [True], [False]],
        0.9, 0.95, "boundary_truncation_mismatch", failures_public, "public_exception",
    )

    for seed in [0, 1, 2]:
        rewards, values, terminated, truncated = _gen_case(seed=seed, T=4, B=2)
        err = _expect_match(
            rewards, values, terminated, truncated,
            0.9, 0.95, "random_public_mismatch", failures_public, "public_exception",
        )
        if err is not None:
            break

    for seed in [10, 11, 12, 13, 14]:
        rewards, values, terminated, truncated = _gen_case(seed=seed, T=6, B=3)
        err = _expect_match(
            rewards, values, terminated, truncated,
            0.97, 0.93, "random_hidden_mismatch", failures_hidden, "hidden_exception",
        )
        if err is not None:
            break

    _expect_match(
        [[1.0], [0.0]],
        [[0.0], [0.0], [1.0]],
        [[False], [True]],
        [[False], [True]],
        0.9, 0.95, "both_true_boundary_mismatch", failures_hidden, "hidden_exception",
    )

    return {
        "public_checks_passed": len(failures_public) == 0,
        "hidden_checks_passed": len(failures_hidden) == 0,
        "failure_messages": failures_public + failures_hidden,
    }


def run_checks_tool() -> RunChecksToolResult:
    return _run_checks()


def _maybe_load_api_key_from_file() -> str | None:
    if os.getenv("DISABLE_KEYFILE"):
        return os.getenv("ANTHROPIC_API_KEY") or None

    existing = os.getenv("ANTHROPIC_API_KEY")
    if existing:
        return existing

    key_path = Path(".secrets/anthropic_key")
    if key_path.exists():
        try:
            key = key_path.read_text(encoding="utf8").strip()
            if key:
                os.environ["ANTHROPIC_API_KEY"] = key
                return key
        except Exception:
            return None
    return None


async def run_doctor() -> None:
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
    max_tokens = int(os.getenv("MAX_TOKENS", "600"))
    max_steps = int(os.getenv("MAX_STEPS", "6"))
    temperature = float(os.getenv("TEMPERATURE", "0.6"))
    allow_offline = os.getenv("ALLOW_OFFLINE_AGENT", "1")

    print(
        json.dumps(
            {
                "mode": "doctor",
                "model": model,
                "max_tokens": max_tokens,
                "max_steps": max_steps,
                "temperature": temperature,
                "allow_offline_agent": allow_offline,
                "disable_keyfile": os.getenv("DISABLE_KEYFILE", "0"),
            },
            indent=2,
        )
    )

    key = _maybe_load_api_key_from_file()
    if key in [None, ""]:
        raise SystemExit(
            "doctor mode requires a real Anthropic API key. Set ANTHROPIC_API_KEY or place it in .secrets/anthropic_key."
        )

    client = AsyncAnthropic(api_key=key)
    started = time.time()
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": "reply with ok"}],
            temperature=0.0,
        )
    except Exception as e:
        raise SystemExit(f"doctor mode failed to call Anthropic: {e}")

    elapsed = time.time() - started
    text_out = "".join(
        [c.text for c in resp.content if getattr(c, "type", "") == "text"]
    ).strip()
    if elapsed <= 1.0:
        raise SystemExit(
            f"doctor mode expected a real network call (>1s). Observed {elapsed:.2f}s."
        )

    print(f"ONLINE doctor ok: '{text_out or 'ok'}' ({elapsed:.2f}s)")


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Any],
    max_steps: int = int(os.getenv("MAX_STEPS", "6")),
    model: str | None = None,
    verbose: bool = True,
    require_online: bool = False,
    allow_offline: bool = True,
    api_key: str | None = None,
) -> dict[str, Any]:
    key = api_key if api_key is not None else _maybe_load_api_key_from_file()
    offline = key in [None, ""] and allow_offline and not require_online

    if offline:
        success_prob = float(os.getenv("OFFLINE_SUCCESS_PROB", "0.35"))
        use_correct = random.random() < success_prob
        code = REFERENCE_ADVANTAGE_CODE if use_correct else BUGGY_ADVANTAGE_CODE
        write_file_tool(code)
        if verbose:
            print(
                "[offline] wrote",
                "correct" if use_correct else "buggy",
                "implementation",
            )
        return {"mode": "offline", "reason": "no_api_key", "fallback_written": True}

    if key in [None, ""]:
        raise RuntimeError("Online run requested but no Anthropic API key is available.")

    if model is None:
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

    client = AsyncAnthropic(api_key=key)
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    max_attempts = int(os.getenv("ANTHROPIC_RETRIES", "3"))
    base_backoff = float(os.getenv("ANTHROPIC_RETRY_BACKOFF", "0.75"))
    last_text_chunks: list[str] = []

    for _ in range(max_steps):
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=MAX_TOKENS,
                    tools=tools,
                    messages=messages,
                    temperature=float(os.getenv("TEMPERATURE", "0.6")),
                )
                break
            except Exception as e:
                if verbose:
                    print(f"[warn] Anthropic call failed (attempt {attempt}/{max_attempts}): {e}")
                if attempt == max_attempts:
                    raise RuntimeError(f"Anthropic API failed after {max_attempts} attempts: {e}")
                await asyncio.sleep(base_backoff * (2 ** (attempt - 1)))

        if response is None:
            raise RuntimeError("Anthropic API returned no response")

        assert response.stop_reason in ["max_tokens", "tool_use", "end_turn"], (
            f"unsupported stop_reason {response.stop_reason}"
        )

        has_tool_use = False
        tool_results = []
        submitted_answer = None

        for content in response.content:
            if content.type == "text":
                text_part = content.text or ""
                last_text_chunks.append(text_part)
                if verbose:
                    print(text_part)
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
                return {"mode": "online", "reason": None, "fallback_written": False}
        else:
            break

    text_blob = "\n".join([t for t in last_text_chunks if t.strip()])
    if "def compute_gae" in text_blob:
        candidate = text_blob
        if "```" in candidate:
            parts = candidate.split("```")
            for i in range(len(parts) - 1):
                block = parts[i + 1]
                if block.lstrip().startswith("python"):
                    candidate = "".join(block.split("\n")[1:])
                    break
        write_file_tool(candidate)
        return {"mode": "online", "reason": "text_fallback", "fallback_written": True}

    return {"mode": "online", "reason": None, "fallback_written": False}


async def run_single_test(
    run_id: int,
    num_runs: int,
    tools,
    tool_handlers,
    verbose: bool = False,
    allow_offline: bool | None = None,
    api_key: str | None = None,
):
    _reset_advantage_file()
    initial_hash = _sha256(Path("advantage.py"))
    prompt = f"""
You are given advantage.py containing a buggy compute_gae implementation. Fix it so all checks pass.

Specification (shared with the grader):
{COMPUTE_GAE_SPEC}

Tools you can call:
1) read_file() -> returns the current contents of advantage.py
2) write_file(content: str) -> overwrites advantage.py (path is ignored)
3) run_checks() -> returns {{"public_checks_passed": bool, "hidden_checks_passed": bool, "failure_messages": [str, ...]}}
4) submit_answer(answer: any) -> finish

Rules:
- Only edit advantage.py. Do not touch main.py or add files.
- Use the tools directly; do not rely on stored state from earlier calls.

Workflow:
read_file() -> write_file(correct_code) -> run_checks() until public and hidden pass -> submit_answer("done")
"""

    started = time.time()
    allow_offline = (
        os.getenv("ALLOW_OFFLINE_AGENT", "1") == "1" if allow_offline is None else allow_offline
    )
    key = api_key if api_key is not None else _maybe_load_api_key_from_file()
    planned_mode = "OFFLINE" if allow_offline and key in [None, ""] else "ONLINE"
    planned_reason = " reason=no_api_key" if planned_mode == "OFFLINE" else ""
    print(f"{planned_mode} start Run {run_id}/{num_runs}{planned_reason}")

    result = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=int(os.getenv("MAX_STEPS", "6")),
        verbose=verbose,
        require_online=not allow_offline,
        allow_offline=allow_offline,
        api_key=key,
    )
    duration = max(time.time() - started, 0.1)

    new_hash = _sha256(Path("advantage.py"))
    offline_reason = result.get("reason") if result.get("mode") == "offline" else None
    mode_label = "OFFLINE" if result.get("mode") == "offline" else "ONLINE"

    if new_hash == initial_hash:
        report: RunChecksToolResult = {
            "public_checks_passed": False,
            "hidden_checks_passed": False,
            "failure_messages": ["no_write_file"],
        }
    else:
        content = Path("advantage.py").read_text(encoding="utf8")
        try:
            ast.parse(content)
        except SyntaxError as e:
            report = {
                "public_checks_passed": False,
                "hidden_checks_passed": False,
                "failure_messages": [
                    f"load_failed_syntaxerror:{e.lineno}:{e.offset}:{e.msg}"
                ],
            }
        else:
            report = _run_checks()

    success = report["public_checks_passed"] and report["hidden_checks_passed"]

    msg = ", ".join(report["failure_messages"][:3]) if not success else ""
    if len(report["failure_messages"]) > 3:
        msg += ", ..."

    reason_suffix = f" reason={offline_reason}" if offline_reason else ""
    if success:
        print(f"{mode_label} ✓ Run {run_id}: SUCCESS ({duration:.1f}s){reason_suffix}")
    else:
        print(
            f"{mode_label} ✗ Run {run_id}: FAILURE ({duration:.1f}s){reason_suffix} - {msg}"
        )

    return run_id, success, report


async def main():
    tools: list[ToolUnionParam] = [
        {
            "name": "read_file",
            "description": "Read advantage.py",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "write_file",
            "description": "Overwrite advantage.py",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "path": {"type": "string", "description": "optional path; defaults to advantage.py"},
                },
                "required": ["content"],
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

    print(f"Using model: {os.getenv('ANTHROPIC_MODEL', 'claude-3-5-haiku-20241022')}")

    tool_handlers = {
        "read_file": read_file_tool,
        "write_file": write_file_tool,
        "run_checks": run_checks_tool,
        "submit_answer": submit_answer_tool,
    }

    args = set(sys.argv[1:])

    if "doctor" in args:
        await run_doctor()
        return

    if "check_current" in args:
        report = _run_checks()
        print(json.dumps(report, indent=2))
        return

    if "check_starter" in args:
        _reset_advantage_file()
        report = _run_checks()
        print(json.dumps(report, indent=2))
        return

    allow_offline = os.getenv("ALLOW_OFFLINE_AGENT", "1") == "1"
    key = _maybe_load_api_key_from_file()
    if not allow_offline and (key in [None, ""]):
        raise SystemExit(
            "Online mode requested but no Anthropic key found. Export ANTHROPIC_API_KEY "
            "or place it in .secrets/anthropic_key, or set ALLOW_OFFLINE_AGENT=1."
        )

    num_runs = int(os.getenv("NUM_RUNS", "10"))
    print(f"Running {num_runs} test iterations sequentially...")
    print("=" * 60)

    results = []
    for i in range(num_runs):
        try:
            result = await run_single_test(
                run_id=i + 1,
                num_runs=num_runs,
                tools=tools,
                tool_handlers=tool_handlers,
                verbose=False,
                allow_offline=allow_offline,
                api_key=key,
            )
        except RuntimeError as e:
            raise SystemExit(str(e))
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
