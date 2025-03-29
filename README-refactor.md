# 项目重构方案

## 目标架构
采用monorepo结构管理三个独立部分：
1. `packages/crawler` - 现有爬虫功能
2. `packages/frontend` - 基于Astro的前端
3. `packages/backend` - 基于Hono的API后端

## 目录结构调整方案

### 新结构
```
.
├── packages/
│   ├── crawler/       # 爬虫组件
│   ├── frontend/      # Astro前端 
│   ├── backend/       # Hono后端API
│   └── core/          # 共享代码(未来提取)
├── docs/              # 文档
├── scripts/           # 项目脚本
└── package.json       # 根项目配置
```

### 具体迁移方案

#### 1. 爬虫部分(crawler)
保留以下目录/文件：
- `lib/` (除前端相关)
- `src/db/raw/` 
- `src/filterWorker.ts`
- `src/worker.ts`
- `test/`
- `deno.json`
- `.gitignore`

需要移除：
- Fresh框架相关文件
- 前端组件(`components/`)
- 静态资源(`static/`)

#### 2. 前端部分(frontend)
全新创建Astro项目，不保留任何现有前端代码

#### 3. 后端部分(backend)
全新创建Hono项目

#### 4. 共享代码(core)
未来可从爬虫中提取以下内容到core package：
- 数据库相关：`lib/db/`
- 消息队列：`lib/mq/`
- 网络请求：`lib/net/`
- 工具函数：`lib/utils/`

## 重构步骤建议

1. 初始化monorepo结构
2. 迁移爬虫代码到`packages/crawler`
3. 创建新的Astro项目在`packages/frontend`
4. 创建新的Hono项目在`packages/backend`
5. 逐步提取共享代码到`packages/core`

## 注意事项
- 机器学习相关代码(`pred/`, `filter/`, `lab/`)保持现状
- 文档(`doc/`)可以迁移到`docs/`目录
- 需要更新CI/CD流程支持monorepo