"""
Guarded in-process execution of model-generated pandas code.

Three layers of defence, in order of importance:

1. **AST validation** (`validate_source`) - the real guard. Rejects imports,
   dunder access, and calls to dangerous builtins *before* anything runs.
2. **Restricted namespace** - `__builtins__` is replaced with a small
   allow-list. `open`, `eval`, `exec`, `__import__`, `compile` are absent.
3. **Wall-clock deadline** - the snippet runs on a worker thread with a
   line-level trace hook that raises once the deadline passes.

Caveat worth knowing: the deadline only fires between *Python* bytecode lines.
A single pandas/numpy call that spends 60s inside C code cannot be interrupted.
That is the accepted trade-off of in-process execution; if you need a hard
kill, move `execute()` behind a subprocess or container.

This module has no LangChain or Django dependency - it is plain Python and is
unit-testable on its own.
"""

from __future__ import annotations

import ast
import io
import sys
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #

#: Names the snippet may never reference.
FORBIDDEN_NAMES = frozenset(
    {
        "__import__", "eval", "exec", "compile", "open", "input",
        "globals", "locals", "vars", "dir", "breakpoint", "help",
        "exit", "quit", "memoryview", "classmethod", "staticmethod",
    }
)

#: Attribute names that expose the object graph / interpreter internals.
FORBIDDEN_ATTR_PREFIXES = ("__",)

#: Explicitly allowed dunder attributes (harmless and commonly useful).
ALLOWED_DUNDER_ATTRS = frozenset({"__len__", "__class__", "__name__", "__doc__"})

#: AST node types that are rejected outright.
FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Lambda,          # not dangerous, but keeps generated code readable
    ast.AsyncFunctionDef,
    ast.Await,
)

#: Builtins the snippet is allowed to call.
_SAFE_BUILTIN_NAMES = (
    "abs all any ascii bin bool bytes callable chr complex dict divmod "
    "enumerate filter float format frozenset hash hex id int "
    "isinstance issubclass iter len list map max min next oct ord pow "
    "print range repr reversed round set slice sorted str sum tuple "
    "type zip True False None Exception ValueError TypeError KeyError "
    "IndexError ZeroDivisionError ArithmeticError AttributeError StopIteration"
).split()


def _guarded_getattr(obj: Any, name: str, *default: Any) -> Any:
    """`getattr` that cannot be used to route around the AST attribute check.

    Without this, `getattr(x, '__subclasses__')` would sail straight past
    `validate_source`, since the dunder never appears as an `ast.Attribute`.
    """
    if not isinstance(name, str):
        raise TypeError("attribute name must be a string")
    if name.startswith("__") and name not in ALLOWED_DUNDER_ATTRS:
        raise AttributeError(f"Access to '{name}' is not allowed.")
    return getattr(obj, name, *default)


def _guarded_hasattr(obj: Any, name: str) -> bool:
    try:
        _guarded_getattr(obj, name)
    except AttributeError:
        return False
    return True


def _safe_builtins() -> dict[str, Any]:
    import builtins

    ns: dict[str, Any] = {}
    for name in _SAFE_BUILTIN_NAMES:
        if hasattr(builtins, name):
            ns[name] = getattr(builtins, name)
    ns["getattr"] = _guarded_getattr
    ns["hasattr"] = _guarded_hasattr
    return ns


class UnsafeCodeError(ValueError):
    """Raised when the snippet fails static validation."""


class ExecutionTimeout(RuntimeError):
    """Raised when the snippet exceeds its wall-clock budget."""


# --------------------------------------------------------------------------- #
# Static validation
# --------------------------------------------------------------------------- #

def validate_source(source: str) -> ast.Module:
    """Parse `source` and reject anything outside the policy.

    Returns the parsed module so the caller does not have to parse twice.
    Raises `UnsafeCodeError` (a ValueError) with a message the agent can read
    and correct itself from.
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:  # surface the position, the model can fix it
        raise UnsafeCodeError(f"SyntaxError: {exc.msg} (line {exc.lineno})") from exc

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise UnsafeCodeError(
                f"{type(node).__name__} is not allowed. "
                "pandas (pd), numpy (np) and matplotlib.pyplot (plt) are already "
                "imported for you - use them directly."
            )

        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"Use of '{node.id}' is not allowed.")

        if isinstance(node, ast.Attribute):
            attr = node.attr
            if (
                attr.startswith(FORBIDDEN_ATTR_PREFIXES)
                and attr not in ALLOWED_DUNDER_ATTRS
            ):
                raise UnsafeCodeError(f"Access to '{attr}' is not allowed.")

    return tree


# --------------------------------------------------------------------------- #
# Deadline enforcement
# --------------------------------------------------------------------------- #

class _Deadline:
    """Line-level trace hook that raises once the budget is spent.

    Used instead of `signal.alarm` because Django serves requests on worker
    threads, where signal handlers cannot be installed.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._expires_at = 0.0

    def __enter__(self) -> "_Deadline":
        import time

        self._expires_at = time.monotonic() + self.seconds
        sys.settrace(self._trace)
        threading.settrace(self._trace)
        return self

    def __exit__(self, *exc: object) -> None:
        sys.settrace(None)
        threading.settrace(None)  # type: ignore[arg-type]

    def _trace(self, frame, event, arg):  # noqa: ANN001 - CPython trace protocol
        import time

        if event == "line" and time.monotonic() > self._expires_at:
            raise ExecutionTimeout(
                f"Execution exceeded {self.seconds:.0f}s. "
                "Work on a smaller slice or aggregate before computing."
            )
        return self._trace


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class ExecResult:
    """Outcome of one snippet."""

    ok: bool
    stdout: str = ""
    value_repr: str = ""
    error: str = ""
    new_names: list[str] = field(default_factory=list)

    def as_text(self, max_chars: int = 6000) -> str:
        """Render for the LLM. Truncated so one wide DataFrame cannot blow the
        context window."""
        if not self.ok:
            return f"ERROR\n{self.error}"

        parts: list[str] = []
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.value_repr.strip():
            parts.append(self.value_repr.rstrip())
        if self.new_names:
            parts.append(f"[variables now available: {', '.join(self.new_names)}]")

        text = "\n".join(parts) if parts else "(no output - use print() to show results)"
        if len(text) > max_chars:
            text = (
                text[:max_chars]
                + f"\n... [truncated at {max_chars} chars - aggregate or "
                ".head() before printing]"
            )
        return text


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #

def execute(
    source: str,
    namespace: dict[str, Any],
    *,
    timeout: float = 30.0,
    max_output_chars: int = 6000,
) -> ExecResult:
    """Validate and run `source` against `namespace` (mutated in place).

    Mirrors notebook semantics: if the final statement is an expression, its
    repr is captured, so the model can end with `df.describe()` and see it.
    """
    try:
        tree = validate_source(source)
    except UnsafeCodeError as exc:
        return ExecResult(ok=False, error=str(exc))

    # Notebook-style: split off a trailing bare expression.
    tail_expr: ast.Expression | None = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop()
        tail_expr = ast.Expression(body=last.value)  # type: ignore[attr-defined]
        ast.fix_missing_locations(tail_expr)

    namespace.setdefault("__builtins__", _safe_builtins())
    before = set(namespace)

    buf_out, buf_err = io.StringIO(), io.StringIO()
    result: dict[str, Any] = {}

    def _run() -> None:
        try:
            with _Deadline(timeout), redirect_stdout(buf_out), redirect_stderr(buf_err):
                if tree.body:
                    exec(compile(tree, "<agent>", "exec"), namespace)  # noqa: S102
                if tail_expr is not None:
                    value = eval(  # noqa: S307
                        compile(tail_expr, "<agent>", "eval"), namespace
                    )
                    if value is not None:
                        result["value"] = value
        except ExecutionTimeout as exc:
            result["error"] = str(exc)
        except BaseException as exc:  # noqa: BLE001 - report, never crash the view
            tb = traceback.format_exception_only(type(exc), exc)
            result["error"] = "".join(tb).strip()

    worker = threading.Thread(target=_run, name="pandas-agent-exec", daemon=True)
    worker.start()
    worker.join(timeout + 5)  # grace period past the trace-hook deadline

    if worker.is_alive():
        return ExecResult(
            ok=False,
            error=(
                f"Execution did not return within {timeout:.0f}s and could not be "
                "interrupted (it is stuck inside a native pandas/numpy call). "
                "Retry on a smaller subset."
            ),
        )

    if "error" in result:
        return ExecResult(ok=False, stdout=buf_out.getvalue(), error=result["error"])

    value_repr = ""
    if "value" in result:
        value_repr = _format_value(result["value"], max_output_chars)

    new_names = sorted(
        n for n in set(namespace) - before if not n.startswith("_")
    )
    return ExecResult(
        ok=True,
        stdout=buf_out.getvalue(),
        value_repr=value_repr,
        new_names=new_names,
    )


def _format_value(value: Any, max_chars: int) -> str:
    """Readable repr for DataFrames/Series without dumping 100k rows."""
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            head = value.head(25)
            text = head.to_string()
            if len(value) > 25:
                text += f"\n... {len(value):,} rows x {value.shape[1]} columns total"
            return text
        if isinstance(value, pd.Series):
            head = value.head(40)
            text = head.to_string()
            if len(value) > 40:
                text += f"\n... {len(value):,} values total"
            return text
    except Exception:  # noqa: BLE001 - formatting must never raise
        pass

    text = repr(value)
    return text if len(text) <= max_chars else text[:max_chars] + " ...[truncated]"
