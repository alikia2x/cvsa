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

The whole CVSA system can be sperate into three different parts:

- Frontend
- API
- Crawler

The frontend is driven by [Astro](https://astro.build/) and is used to display the final CVSA page. The API is driven by
[Hono](https://hono.dev) and is used to query the database and provide REST/GraphQL APIs that can be called by out
website, applications, or third parties. The crawler is our automatic data collector, used to automatically collect new
songs from bilibili, track their statistics, etc.

### Crawler

Automation is the biggest highlight of CVSA's technical design. To achieve this, we use a message queue powered by
[BullMQ](https://bullmq.io/) to concurrently process various tasks in the data collection life cycle.
