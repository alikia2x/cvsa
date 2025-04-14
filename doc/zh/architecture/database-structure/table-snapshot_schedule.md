# snapshot\_schedule 表

该表用于记录视频快照任务的调度信息。

### 字段说明

| 字段名           | 类型                         | 是否为空 | 默认值                                   | 描述           |
| ------------- | -------------------------- | ---- | ------------------------------------- | ------------ |
| `id`          | `bigint`                   | 否    | `nextval('snapshot_schedule_id_seq')` | 主键，自增ID      |
| `aid`         | `bigint`                   | 否    | 无                                     | 哔哩哔哩视频的 AV 号 |
| `type`        | `text`                     | 是    | 无                                     | 快照类型。        |
| `created_at`  | `timestamp with time zone` | 否    | `CURRENT_TIMESTAMP`                   | 记录创建时间       |
| `started_at`  | `timestamp with time zone` | 是    | 无                                     | 计划开始拍摄快照的时间  |
| `finished_at` | `timestamp with time zone` | 是    | 无                                     | 快照任务完成的时间    |
| `status`      | `text`                     | 否    | `'pending'`                           | 快照任务状态。      |

### 字段取值说明（待补充）

#### `type` 字段

用于标识快照的类型，例如是定期存档、成就节点、首次收录等。

* `archive`：每隔一段时间内，对`bilibili_metadata`表中所有视频的定期快照。
* `milestone`：监测到曲目即将达成成就（殿堂/传说/神话）时，将会调度该类型的快照任务。
* `new`：新观测到歌曲时，会在最长48小时内持续追踪其初始播放量增长趋势。
* `normal`：对于所有`songs`表内的曲目，根据播放量增长速度，以动态间隔（6-72小时）定期进行的快照。

#### `status` 字段

用于标识快照任务的当前状态。

* `completed`：快照任务已经完成
* `failed`：快照任务因不明原因失败
* `no_proxy`：快照任务被执行，但当前没有代理可用于拍摄快照
* `pending`：快照任务已经被调度，但尚未开始执行
* `processing`：正在获取快照
* `timeout`：快照任务在一定时间内没有被响应，因此被丢弃
* `bili_error`: 哔哩哔哩返回了一个表示请求失败的状态码

### 备注

* 此表中的 `started_at` 字段为计划中的快照开始时间，实际执行时间可能与其略有偏差，具体执行记录可结合其他日志或任务表查看。
* 每个 av 号在可以同时存在多个不同类型的快照任务处于 pending 状态，但对于同一种类型，只允许一个pending任务同时存在。
