---
description: 关于VideoTagsQueue队列的信息。
---

# VideoTagsQueue队列

### 任务

视频标签队列包含两个任务：`getVideoTags`和`getVideosTags`。前者用于获取视频的标签，后者负责调度前者。

### 返回值

两个任务的返回值遵循以下表格：

<table><thead><tr><th width="168">返回值</th><th>描述</th></tr></thead><tbody><tr><td>0</td><td>在 <code>getVideoTags</code> 中：标签成功获取<br>在 <code>getVideosTags</code> 中：所有无标签视频的相应任务已成功排队。</td></tr><tr><td>1</td><td>在 <code>getVideoTags</code> 中：任务期间发生 <code>fetch</code> 错误</td></tr><tr><td>2</td><td>在 <code>getVideoTags</code> 中：已达到 NetScheduler 设置的速率限制</td></tr><tr><td>3</td><td>在 <code>getVideoTags</code> 中：未在任务数据中提供帮助</td></tr><tr><td>4</td><td>在 <code>getVideosTags</code> 中：没有视频的 `tags` 为 NULL</td></tr><tr><td>1xx</td><td>在 <code>getVideosTags</code> 中：队列中的任务数量超过了限制，因此 <code>getVideosTags</code> 停止添加任务。<code>xx</code> 是在执行期间添加到队列的任务数量。</td></tr></tbody></table>
