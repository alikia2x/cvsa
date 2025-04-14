# 数据库结构

CVSA 使用 [PostgreSQL](https://www.postgresql.org/) 作为数据库。

CVSA 设计了两个数据库，`cvsa_main` 和 `cvsa_cred`。前者用于存储可公开的数据，而后者则存储用户相关的个人信息（如登录凭据、账户管理信息等）。

CVSA 的所有公开数据（不包括用户的个人数据）都存储在名为 `cvsa_main` 的数据库中，该数据库包含以下表：

* songs：存储歌曲的主要信息。
* bilibili\_user：存储哔哩哔哩 UP主 的元信息。
* bilibili\_metadata：我们收录的哔哩哔哩所有视频的元数据。
* labelling\_result：包含由我们的机器学习模型标记的 `bilibili_metadata` 中视频的标签。
* latest\_video\_snapshot：存储视频最新的快照。
* video\_snapshot：存储视频的快照，包括特定时间下视频的统计信息（播放量、点赞数等）。
* snapshot\_schedule：视频快照的规划信息，为辅助表。

> **快照：**
>
> 我们定期采集哔哩哔哩视频的播放量、点赞收藏数等统计信息，在一个给定时间点下某支视频的统计数据即为该视频的一个快照。



