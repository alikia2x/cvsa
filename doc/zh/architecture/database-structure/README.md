# 数据库结构

CVSA 使用 [PostgreSQL](https://www.postgresql.org/) 作为数据库。

CVSA 的所有公开数据（不包括用户的个人数据）都存储在名为 `cvsa_main` 的数据库中，该数据库包含以下表：

* songs：存储歌曲的主要信息
* bili\_user：存储 Bilibili 用户信息快照
* all\_data：[分区 30](../../about/scope-of-inclusion.md#vocaloiduatu-fen-qu) 中所有视频的元数据。
* labelling\_result：包含由我们的 AI 系统 标记的 `all_data` 中视频的标签。
