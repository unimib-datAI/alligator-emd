import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro"
          >
            Get Started →
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            href="https://github.com/unimib-datAI/alligator-emd"
          >
            GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}

const features = [
  {
    title: 'Entity Linking',
    description:
      'Link cells in CSV tables to Wikidata entities using a two-stage ML ranking pipeline (rank → rerank).',
  },
  {
    title: 'Semantic Annotations',
    description:
      'Produces CEA (cell entity), CTA (column type), and CPA (column property) annotations conforming to SemTab standards.',
  },
  {
    title: 'Flexible API',
    description:
      'Use as a Python library, CLI tool, or REST API via the included FastAPI backend. Docker Compose ready.',
  },
];

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description={siteConfig.tagline}
    >
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              {features.map(({ title, description }) => (
                <div key={title} className={clsx('col col--4')}>
                  <div className="text--center padding-horiz--md padding-vert--md">
                    <h3>{title}</h3>
                    <p>{description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
