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

# 概览

整个CVSA项目分为三个组件：**crawler**, **frontend** 和 **backend。**

### **crawler**

位于项目目录`packages/crawler` 下，它负责以下工作：

- 抓取新的视频并收录作品
- 持续监控视频的播放量等统计信息

整个 crawler 由 BullMQ 消息队列驱动，使用 Redis 和 PostgreSQL 管理状态。
