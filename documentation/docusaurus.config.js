// @ts-check
const { themes: prismThemes } = require("prism-react-renderer");

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "Alligator",
  tagline: "Entity Linking for Tabular Data",
  url: "https://unimib-datAI.github.io",
  baseUrl: "/alligator-emd/",
  organizationName: "unimib-datAI",
  projectName: "alligator-emd",
  trailingSlash: false,
  onBrokenLinks: "throw",
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: "warn",
    },
  },
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve("./sidebars.js"),
          editUrl:
            "https://github.com/unimib-datAI/alligator-emd/tree/main/documentation/",
        },
        blog: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: "Alligator",
        logo: {
          alt: "Alligator Logo",
          src: "img/logo.svg",
        },
        items: [
          {
            type: "docSidebar",
            sidebarId: "docsSidebar",
            position: "left",
            label: "Docs",
          },
          {
            href: "https://github.com/unimib-datAI/alligator-emd",
            label: "GitHub",
            position: "right",
          },
        ],
      },
      footer: {
        style: "dark",
        links: [
          {
            title: "Docs",
            items: [
              { label: "Introduction", to: "/docs/intro" },
              { label: "Quick Start", to: "/docs/quick-start" },
              { label: "CLI Reference", to: "/docs/cli" },
            ],
          },
          {
            title: "Links",
            items: [
              {
                label: "GitHub",
                href: "https://github.com/unimib-datAI/alligator-emd",
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} unimib-datAI. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ["python", "bash", "json"],
      },
    }),
};

module.exports = config;
