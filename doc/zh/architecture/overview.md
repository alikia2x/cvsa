# 概览

CVSA 是一个 [monorepo](https://en.wikipedia.org/wiki/Monorepo) 代码库，使用 [Deno workspace](https://docs.deno.com/runtime/fundamentals/workspaces/) 作为monorepo管理工具，TypeScript 是主要的开发语言。

**项目结构：**

```
cvsa
├── deno.json
├── ml
│   ├── filter
│   ├── lab
│   └── pred
├── packages
│   ├── backend
│   ├── core
│   ├── crawler
│   └── frontend
└── README.md
```

**其中， `packages` 为 monorepo 主要的根目录，包含 CVSA 主要的程序逻辑**

* **`backend`**：这个模块包含使用 [Hono](https://hono.dev/) 框架构建的服务器端逻辑。它负责与数据库交互并通过 REST 和 GraphQL API 公开数据，供前端网站、应用和第三方使用。
* **`frontend`**：中V档案馆的网站是 [Astro](https://astro.build/) 驱动的。这个模块包含完整的 Astro 前端项目。
* **`crawler`**：这个模块包含中V档案馆的自动数据收集系统。它旨在自动发现和收集来自哔哩哔哩的新歌曲数据，以及跟踪相关统计数据（如播放量信息）。
* **`core`**：这个模块内包含可重用和通用的代码。

`ml` 为机器学习相关包，参见
