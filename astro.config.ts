import { defineConfig } from 'astro/config';

import expressiveCode from 'astro-expressive-code';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import spectre from './package/src';

import node from '@astrojs/node';
import { spectreDark } from './src/ec-theme';

// https://astro.build/config
export default defineConfig({
  site: 'https://pinakinchoudhary.com',
  output: 'static',
  integrations: [
    expressiveCode({
      themes: ['github-dark'],
    }),
    mdx(),
    sitemap(),
    spectre({
      name: 'Pinakin Choudhary',
      openGraph: {
        home: {
          title: 'About Me',
          description: 'My homepage and portfolio.'
        },
        blog: {
          title: 'Blog',
          description: 'My personal research and development blog.'
        },
        projects: {
          title: 'Projects'
        }
      },
      giscus: {
        repository: 'louisescher/spectre',
        repositoryId: 'R_kgDONjm3ig',
        category: 'General',
        categoryId: 'DIC_kwDONjm3is4ClmBF',
        mapping: 'pathname',
        strict: true,
        reactionsEnabled: true,
        emitMetadata: false,
        lang: 'en',
      }
    })
  ],
  adapter: node({
    mode: 'standalone'
  })
});