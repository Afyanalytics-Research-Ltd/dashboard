"""
Validate GraphQL documents against the Model Studio allowlist.

Root fields are collected from every operation in the document, following
inline fragments and named fragment spreads at the root, so an operation
cannot hide a blocked field behind a fragment.
"""

from dataclasses import dataclass

from graphql import GraphQLError, parse
from graphql.language import (
    FieldNode, FragmentDefinitionNode, FragmentSpreadNode, InlineFragmentNode, OperationDefinitionNode,
)
from graphql.utilities import value_from_ast_untyped

from .operations import ALLOWED_MUTATIONS, ALLOWED_QUERIES

INTROSPECTION = {'__schema', '__type', '__typename'}
MAX_DOCUMENT_BYTES = 256 * 1024


@dataclass
class Checked:
    ok: bool
    error: str = ''
    mutations: tuple[str, ...] = ()
    queries: tuple[str, ...] = ()

    @property
    def read_only(self) -> bool:
        return not self.mutations


def _root_field_nodes(selection_set, fragments, seen=frozenset()):
    for sel in selection_set.selections:
        if isinstance(sel, FieldNode):
            yield sel
        elif isinstance(sel, InlineFragmentNode):
            yield from _root_field_nodes(sel.selection_set, fragments, seen)
        elif isinstance(sel, FragmentSpreadNode):
            name = sel.name.value
            if name in seen or name not in fragments:
                continue
            yield from _root_field_nodes(fragments[name].selection_set, fragments, seen | {name})


def mutation_inputs(document: str, variables: dict | None) -> list[tuple[str, dict]]:
    """
    [(mutation_name, resolved arguments)] for every mutation root field, with
    variables substituted. Best effort: unparsable documents yield [].
    """
    try:
        doc = parse(document)
    except GraphQLError:
        return []
    fragments = {d.name.value: d for d in doc.definitions if isinstance(d, FragmentDefinitionNode)}
    out = []
    for op in doc.definitions:
        if not isinstance(op, OperationDefinitionNode) or op.operation.value != 'mutation':
            continue
        for node in _root_field_nodes(op.selection_set, fragments):
            args = {a.name.value: value_from_ast_untyped(a.value, variables or {}) for a in node.arguments or []}
            out.append((node.name.value, args))
    return out


def _root_fields(selection_set, fragments, seen=frozenset()):
    for sel in selection_set.selections:
        if isinstance(sel, FieldNode):
            yield sel.name.value
        elif isinstance(sel, InlineFragmentNode):
            yield from _root_fields(sel.selection_set, fragments, seen)
        elif isinstance(sel, FragmentSpreadNode):
            name = sel.name.value
            if name in seen or name not in fragments:
                continue
            yield from _root_fields(fragments[name].selection_set, fragments, seen | {name})


def check_document(document: str) -> Checked:
    if not isinstance(document, str) or not document.strip():
        return Checked(False, 'A GraphQL "query" string is required.')
    if len(document.encode()) > MAX_DOCUMENT_BYTES:
        return Checked(False, 'GraphQL document is too large.')
    try:
        doc = parse(document)
    except GraphQLError as exc:
        return Checked(False, f'GraphQL syntax error: {exc.message}')

    fragments = {d.name.value: d for d in doc.definitions if isinstance(d, FragmentDefinitionNode)}
    operations = [d for d in doc.definitions if isinstance(d, OperationDefinitionNode)]
    if not operations:
        return Checked(False, 'The document contains no operation.')
    queries, mutations = [], []
    for op in operations:
        kind = op.operation.value
        if kind == 'subscription':
            return Checked(False, 'GraphQL subscriptions are not supported.')
        allowed = ALLOWED_MUTATIONS if kind == 'mutation' else ALLOWED_QUERIES
        target = mutations if kind == 'mutation' else queries
        for field in _root_fields(op.selection_set, fragments):
            if field in INTROSPECTION and kind == 'query':
                continue
            if field not in allowed:
                return Checked(False, f'The {kind} "{field}" is not available through Model Studio.')
            target.append(field)
    return Checked(True, mutations=tuple(mutations), queries=tuple(queries))
