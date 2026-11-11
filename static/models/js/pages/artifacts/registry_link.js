// "Link to registry" dialog: attach a model version to a registered model (artifact portfolio)
// and optionally give it aliases there (staging / production …).

import { h, icon, mount } from '../../../../ui/js/dom.js';
import { modal, toast, buildForm, skeleton } from '../../../../ui/js/ui.js';
import { gql } from '../../core/api.js';
import { urls } from '../../core/urls.js';
import { navigate } from '../../../../ui/js/router.js';

export const REGISTRY_PROJECT = 'model-registry';

/** Registered models (portfolios) visible in an entity. */
export async function loadPortfolios(entity, { signal } = {}) {
  const d = await gql(`query Ports($e: String!) { entity(name: $e) {
    artifactCollections(first: 200, collectionTypes: [PORTFOLIO]) { edges { node { id name description createdAt updatedAt project { name entityName }
      aliases(first: 30) { edges { node { alias } } }
      artifacts(first: 1) { totalCount } } } } } }`, { e: entity }, { signal });
  return (d.entity?.artifactCollections?.edges || []).map((e) => e.node);
}

export function linkToRegistry({ entity, project, collection, artifact, onDone }) {
  const listEl = h('div', skeleton(2));
  const form = buildForm([
    { name: 'portfolio', label: 'Registered model name', required: true, value: collection.replace(/[^\w.-]+/g, '-'), help: 'Pick an existing registered model below or type a new name to create one.' },
    { name: 'project', label: 'Registry project', value: REGISTRY_PROJECT, required: true, help: `W&B keeps registered models in a project; “${REGISTRY_PROJECT}” is the convention.` },
    { name: 'aliases', label: 'Aliases in the registry', type: 'tags', value: [], placeholder: 'staging, production', help: 'Optional. Stable names such as staging or production.' },
  ]);
  const nameInput = form.el.querySelector('input');
  loadPortfolios(entity).then((ports) => {
    mount(listEl, ports.length
      ? h('div.ma-port-picks', h('div.small.fw-semibold.text-muted.mb-1', 'Existing registered models'),
        ports.map((p) => h('button.ma-port-pick', { type: 'button', onclick: () => {
          nameInput.value = p.name;
          form.el.querySelectorAll('input')[1].value = p.project?.name || REGISTRY_PROJECT;
        } }, icon('box-seam'), h('span', p.name), h('span.text-muted.small', `${p.artifacts?.totalCount ?? 0} versions · ${p.project?.name}`))))
      : h('div.small.text-muted', 'No registered models yet — this will create the first one.'));
  }).catch(() => mount(listEl));
  return new Promise((resolve) => {
    let done = false;
    modal({
      title: `Link ${collection}:${artifact.versionLabel} to the registry`, icon: 'box-arrow-in-up-right', size: 'lg',
      body: h('div', h('div.af-callout.af-callout-info.mb-3', icon('info-circle'),
        h('span', 'Linking does not copy files: the registered model points at this exact version, so its lineage and metrics stay connected.')),
      form.el, h('div.mt-3', listEl)),
      actions: [
        { label: 'Cancel' },
        {
          label: 'Link version', variant: 'primary', icon: 'link-45deg',
          onClick: async (close) => {
            const v = form.values();
            if (!/^[\w.-]+$/.test(v.portfolio)) throw new Error('Registered model names may contain letters, digits, “-”, “_” and “.”.');
            const aliases = (v.aliases || []).map((alias) => ({ alias, artifactCollectionName: v.portfolio }));
            const d = await gql(`mutation Link($input: LinkArtifactInput!) { linkArtifact(input: $input) { versionIndex artifactCollectionID } }`,
              { input: { artifactID: artifact.id, artifactPortfolioName: v.portfolio, entityName: entity, projectName: v.project, aliases } });
            done = true;
            close();
            toast(`Linked as ${v.portfolio}:v${d.linkArtifact?.versionIndex ?? '?'}`, 'success', {
              title: 'Added to registry',
              action: { label: 'Open in registry →', onClick: () => navigate(urls.registry({ model: v.portfolio })) },
            });
            onDone?.();
            resolve(d.linkArtifact);
          },
        },
      ],
      onClose: () => { if (!done) resolve(null); },
    });
  });
}

export async function unlinkFromRegistry(artifactId, portfolioId) {
  await gql(`mutation Unlink($input: UnlinkArtifactInput!) { unlinkArtifact(input: $input) { artifactID } }`, { input: { artifactID: artifactId, artifactPortfolioID: portfolioId } });
}
