// Live GraphQL schema (introspected through the proxy) + query generation for the explorer.

import { gql } from '../../core/api.js';

// Kept in sync with wandb_ui/operations.py BLOCKED (cloud billing, sandboxes, Slurm, org-destroying).
export const BLOCKED = new Set(['ackSlurmCommand', 'attachCustomerPaymentMethod', 'cancelCustomerSubscription', 'cancelPaygoSubscriptions', 'checkoutPreview',
  'claimSlurmCommands', 'contactUsForComputeHours', 'contactUsForUserLedTrial', 'coupon', 'createAcademicOrReverseTrial', 'createAnonymousEntity',
  'createCustomerSubscription', 'createCustomerTrial', 'createHourOverageSubscription', 'createInferenceSubscription', 'createLocalLicenseOrganization',
  'createNewsletterSubscription', 'createPricingPlanUpdatePR', 'createSandbox', 'createSandboxPermission', 'createSandboxesSubscription',
  'createStorageSubscription', 'deleteOrganization', 'deleteOrganizationSubscription', 'detachCustomerPaymentMethod', 'disableOldTrackingSubscription',
  'ensureSandboxesOverageSubscription', 'ensureWbagentOverageSubscription', 'failSlurmCommand', 'failedPaymentBanners', 'generateLocalLicense',
  'grantPersonalSecretToWBAgent', 'heartbeatSandbox', 'makeOrgMembersNoAccessModelsSeat', 'migrateOldHourOverageSubscription', 'migrateOldStorageSubscription',
  'migrateOrPurgePersonalEntity', 'newsletterSubscriptions', 'plans', 'purgeUser', 'rampManager', 'reactivateCustomerSubscription', 'recordSlurmCommandError',
  'resumePaygoSubscriptions', 'retryStripeOpenInvoicesByOrganization', 'revokePersonalSecretFromWBAgent', 'sandboxReadPermission', 'setSandboxEnvVars',
  'submitFCNews', 'submitSlurmCommand', 'terminateSandbox', 'unlinkNvidiaAccount', 'updateCustomerDefaultPaymentMethod', 'updateCustomerPaymentMethod',
  'updateModelsSeat', 'updateOrganizationBillingUser', 'updateOrganizationSubscription', 'updateOrganizationSubscriptionSeats', 'updateRampThreshold',
  'updateSlurmDaemonHeartbeat', 'updateSubscription', 'upgradeCustomerSubscription', 'upgradeCustomerToProPlan', 'upgradeProCustomer',
  'upgradeTrialSubscription', 'weaveSavingsCalculator']);

const TYPE_REF = 'kind name ofType { kind name ofType { kind name ofType { kind name ofType { kind name } } } }';
const INTROSPECTION = `query Introspection { __schema { queryType { name } mutationType { name }
  types { kind name description
    fields(includeDeprecated: true) { name description isDeprecated deprecationReason args { name description defaultValue type { ${TYPE_REF} } } type { ${TYPE_REF} } }
    inputFields { name description defaultValue type { ${TYPE_REF} } }
    enumValues(includeDeprecated: false) { name description }
    possibleTypes { name } interfaces { name } } } }`;

let cached = null;
const CACHE_KEY = 'ms:gqlschema:v1';

/** Load (and cache for the session) the schema: {types: Map, query: Type, mutation: Type}. */
export async function loadSchema({ fresh = false } = {}) {
  if (cached && !fresh) return cached;
  let raw = null;
  if (!fresh) {
    try { raw = JSON.parse(sessionStorage.getItem(CACHE_KEY) || 'null'); } catch { raw = null; }
  }
  if (!raw) {
    raw = (await gql(INTROSPECTION)).__schema;
    try { sessionStorage.setItem(CACHE_KEY, JSON.stringify(raw)); } catch { /* quota: fine */ }
  }
  const types = new Map(raw.types.map((t) => [t.name, t]));
  cached = { types, query: types.get(raw.queryType.name), mutation: types.get(raw.mutationType?.name), raw };
  return cached;
}

/** Strip NON_NULL/LIST wrappers. */
export function named(t) { while (t && (t.kind === 'NON_NULL' || t.kind === 'LIST')) t = t.ofType; return t; }
/** "[String!]!" style signature. */
export function sig(t) {
  if (!t) return '';
  if (t.kind === 'NON_NULL') return `${sig(t.ofType)}!`;
  if (t.kind === 'LIST') return `[${sig(t.ofType)}]`;
  return t.name;
}
export const isLeaf = (schema, t) => ['SCALAR', 'ENUM'].includes(schema.types.get(named(t).name)?.kind);
export const isList = (t) => (t?.kind === 'NON_NULL' ? isList(t.ofType) : t?.kind === 'LIST');
export const isRequired = (t) => t?.kind === 'NON_NULL';

const SKIP_DEFAULT = /^(uploadUrl|uploadHeaders|uploadMultipartUrls|photoUploadUrl|stripe|apiKey$|secret|password)/i;
const PREFERRED = ['id', 'name', 'displayName', 'username', 'email', 'state', 'createdAt', 'updatedAt', 'description', 'entityName', 'success', 'totalRuns'];

/** Sensible default selection for a type: a handful of cheap scalar fields (no-arg). */
export function defaultSelection(schema, typeName, max = 6) {
  const t = schema.types.get(typeName);
  if (!t?.fields) return [];
  const scalars = t.fields.filter((f) => !f.isDeprecated && !f.args.some((a) => isRequired(a.type)) && isLeaf(schema, f.type) && !SKIP_DEFAULT.test(f.name));
  const ordered = [...scalars].sort((a, b) => {
    const ia = PREFERRED.indexOf(a.name);
    const ib = PREFERRED.indexOf(b.name);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  return ordered.slice(0, max).map((f) => f.name);
}

/**
 * Build query text from a selection model.
 *  op: 'query' | 'mutation'; field: root field def; selection: Tree where tree = { fieldName: subtree|true, '...on X': subtree }
 *  args: { argName: value-or-undefined } (only set args become $variables)
 */
export function buildDocument(schema, op, field, args, selection) {
  const used = field.args.filter((a) => args[a.name] !== undefined && args[a.name] !== '');
  const varDefs = used.map((a) => `$${a.name}: ${sig(a.type)}`).join(', ');
  const argText = used.map((a) => `${a.name}: $${a.name}`).join(', ');
  const body = isLeaf(schema, field.type) ? '' : ` ${renderSelection(selection, 2)}`;
  const opName = `${field.name[0].toUpperCase()}${field.name.slice(1)}`;
  const text = `${op} ${opName}${varDefs ? `(${varDefs})` : ''} {\n  ${field.name}${argText ? `(${argText})` : ''}${body}\n}`;
  const variables = Object.fromEntries(used.map((a) => [a.name, args[a.name]]));
  return { text, variables };
}

function renderSelection(tree, depth) {
  const pad = '  '.repeat(depth);
  const entries = Object.entries(tree || {});
  if (!entries.length) return '{ __typename }';
  const lines = entries.map(([k, v]) => (v === true ? `${pad}${k}` : `${pad}${k} ${renderSelection(v, depth + 1)}`));
  return `{\n${lines.join('\n')}\n${'  '.repeat(depth - 1)}}`;
}

/** Parse a user-typed argument value according to its GraphQL type. */
export function coerce(schema, type, raw) {
  if (raw === '' || raw == null) return undefined;
  const t = named(type);
  const kind = schema.types.get(t.name)?.kind;
  if (isList(type) || kind === 'INPUT_OBJECT') {
    if (typeof raw !== 'string') return raw;
    try { return JSON.parse(raw); } catch { if (isList(type)) return raw.split(',').map((s) => s.trim()).filter(Boolean); throw new Error(`${sig(type)} needs JSON`); }
  }
  if (t.name === 'Int' || t.name === 'Int64') return Number.parseInt(raw, 10);
  if (t.name === 'Float') return Number(raw);
  if (t.name === 'Boolean') return raw === true || raw === 'true';
  return raw;
}

/** Skeleton JSON for an input object type (required fields first), for prefilled editors. */
export function inputSkeleton(schema, typeName, depth = 0) {
  const t = schema.types.get(typeName);
  if (!t?.inputFields || depth > 2) return {};
  const out = {};
  const fields = [...t.inputFields].sort((a, b) => Number(isRequired(b.type)) - Number(isRequired(a.type)));
  for (const f of fields) {
    if (!isRequired(f.type) && depth > 0) continue;
    if (f.name === 'clientMutationId') continue;
    const n = named(f.type);
    const k = schema.types.get(n.name)?.kind;
    if (isList(f.type)) out[f.name] = [];
    else if (k === 'INPUT_OBJECT') out[f.name] = inputSkeleton(schema, n.name, depth + 1);
    else if (k === 'ENUM') out[f.name] = schema.types.get(n.name).enumValues?.[0]?.name ?? null;
    else if (n.name === 'Boolean') out[f.name] = false;
    else if (/Int|Float/.test(n.name)) out[f.name] = 0;
    else out[f.name] = isRequired(f.type) ? '' : null;
  }
  return out;
}
