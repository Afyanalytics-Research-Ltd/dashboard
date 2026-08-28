"""
Safe arithmetic expression evaluator for derived metrics.

Used to compute a ratio/difference/sum from 2+ base measures already
returned in the SAME Cube query row (see agents/derived_metrics.py) —
NEVER by combining separately-queried, pre-aggregated result sets, which
would silently produce wrong numbers under any grouping.

Whitelisted AST only — no eval()/exec() anywhere, no function calls, no
attribute/subscript access, no comprehensions. A hostile or hallucinated
expression string can only ever fail loudly here, never execute arbitrary
code.
"""

from __future__ import annotations

import ast


class ExprEvalError(Exception):
    pass


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARYOPS = (ast.USub, ast.UAdd)


def _check_node(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _check_node(node.body)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise ExprEvalError(f"operator {type(node.op).__name__} not allowed")
        _check_node(node.left)
        _check_node(node.right)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise ExprEvalError(f"unary operator {type(node.op).__name__} not allowed")
        _check_node(node.operand)
        return
    if isinstance(node, ast.Name):
        return  # existence in `variables` is checked at eval time, not parse time
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise ExprEvalError("only numeric literals allowed")
        return
    raise ExprEvalError(f"node type {type(node).__name__} not allowed")


def _eval_node(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ExprEvalError("division by zero")
            return left / right
        raise ExprEvalError(f"unhandled operator {type(node.op).__name__}")
    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, variables)
        return -val if isinstance(node.op, ast.USub) else +val
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ExprEvalError(f"unbound variable '{node.id}'")
        return variables[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    raise ExprEvalError(f"cannot evaluate node type {type(node).__name__}")


def evaluate(expression: str, variables: dict[str, float]) -> float:
    """
    Evaluate a whitelisted arithmetic expression string (+, -, *, /, unary
    +/-, numeric literals, and names bound in `variables`) — nothing else.
    Raises ExprEvalError on anything not on that whitelist, on an unbound
    variable, or on division by zero.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExprEvalError(f"invalid expression syntax: {exc}") from exc
    _check_node(tree)
    return _eval_node(tree.body, variables)
