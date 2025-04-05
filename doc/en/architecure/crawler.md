# Crawler

A central aspect of CVSA's technical design is its emphasis on automation. The data collection process within the `crawler` is orchestrated using a message queue powered by [BullMQ](https://bullmq.io/). This enables concurrent processing of various tasks involved in the data lifecycle. State management and data persistence are handled by a combination of Redis for caching and real-time data, and PostgreSQL as the primary database.

