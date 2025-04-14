# Crawler 模块介绍

在中V档案馆的技术架构中，自动化是核心设计理念。`crawler` 模块负责整个数据采集流程，通过 [BullMQ](https://bullmq.io/) 实现任务的消息队列管理，支持高并发地处理多个采集任务。

系统的数据存储与状态管理采用了 Redis（用于缓存和实时数据）与 PostgreSQL（作为主数据库）的组合方式，确保了稳定性与高效性。

***

### 模块结构概览

#### `crawler/db` —— 数据库操作模块

负责与数据库的交互，提供创建、更新、查询等功能。

* `init.ts`：初始化 PostgreSQL 连接池。
* `redis.ts`：配置 Redis 客户端。
* `withConnection.ts`：导出 `withDatabaseConnection` 函数，用于包装数据库操作函数，提供数据库上下文。
* 其他文件：每个文件对应数据库中的一张表，封装了该表的操作逻辑。

#### `crawler/ml` —— 机器学习模块

负责与机器学习模型相关的处理逻辑，主要用于视频内容的文本分类。

* `manager.ts`：定义了一个模型管理基类 `AIManager`。
* `akari.ts`：实现了用于筛选歌曲视频的分类模型 `AkariProto`，继承自 `AIManager`。

#### `crawler/mq` —— 消息队列模块

整合 BullMQ，实现任务调度和异步处理。

**`crawler/mq/exec`**

该目录下包含了各类任务的处理函数。虽然这些函数并非 BullMQ 所直接定义的“worker”，但在文档中我们仍将其统一称为 **worker**（例如 `getVideoInfoWorker`、`takeBulkSnapshotForVideosWorker`）。

> **说明：**
>
> * `crawler/mq/exec` 中的函数称为 **worker**。
> * `crawler/mq/workers` 中的函数我们称为 **BullMQ worker**。

**架构设计说明：**\
由于 BullMQ 设计上每个队列只能有一个处理函数，我们通过 `switch` 语句在一个 worker 中区分并路由不同的任务类型，将其分发给相应的执行函数。

**`crawler/mq/workers`**

这个目录定义了真正的 BullMQ worker，用于消费对应队列中的任务，并调用具体的执行逻辑。

**`crawler/mq/task`**

为了保持 worker 函数的简洁与可维护性，部分复杂逻辑被抽离成独立的“任务（task）”函数，集中放在这个目录中。

#### `crawler/net` —— 网络请求模块

该模块用于与外部系统通信，负责所有网络请求的封装和管理。核心是 `net/delegate.ts` 中定义的 `NetworkDelegate` 类。

**`crawler/net/delegate.ts`**

这是我们进行大规模请求的主要实现，支持以下功能：

* 基于任务类型和代理的限速策略
* 结合 serverless 架构，根据策略动态切换请求来源 IP

#### `crawler/utils` —— 工具函数模块

存放项目中通用的工具函数，供各模块调用。

#### `crawler/src` —— 主程序入口

该目录包含 crawler 的启动脚本。我们使用 [concurrently](https://www.npmjs.com/package/concurrently) 同时运行多个任务文件，实现并行处理。
