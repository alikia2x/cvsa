---
layout:
  title:
    visible: true
  description:
    visible: false
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
---

# Overview

The CVSA is a [monorepo](https://en.wikipedia.org/wiki/Monorepo) codebase, mainly using TypeScript as the development language. With [Deno workspace](https://docs.deno.com/runtime/fundamentals/workspaces/), the major part of the codebase is under `packages/`.&#x20;

**Project structure:**

```
cvsa
├── deno.json
├── packages
│   ├── backend
│   ├── core
│   ├── crawler
│   └── frontend
└── README.md
```

**Package Breakdown:**

- **`backend`**: This package houses the server-side logic, built with the [Hono](https://hono.dev/) web framework. It's responsible for interacting with the database and exposing data through REST and GraphQL APIs for consumption by the frontend, internal applications, and third-party developers.
- **`frontend`**: The user-facing web interface of CVSA is developed using [Astro](https://astro.build/). This package handles the presentation layer, displaying information fetched from the database.
- **`crawler`**: This automated data collection system is a key component of CVSA. It's designed to automatically discover and gather new song data from bilibili, as well as track relevant statistics over time.
- **`core`**: This package contains reusable and generic code that is utilized across multiple workspaces within the CVSA monorepo.
