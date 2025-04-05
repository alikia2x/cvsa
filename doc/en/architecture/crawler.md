# Crawler

Automation is at the core of CVSA’s technical architecture. The `crawler` is built to efficiently orchestrate data collection tasks using a message queue system powered by [BullMQ](https://bullmq.io/). This design enables concurrent processing across multiple stages of the data collection lifecycle. 

State management and data persistence are handled using a combination of Redis (for caching and real-time data) and PostgreSQL (as the primary database).

## `crawler/db`

This module handles all database interactions for the crawler, including creation, updates, and data retrieval.

- `init.ts`: Initializes the PostgreSQL connection pool.
- `redis.ts`: Sets up the Redis client.
- `withConnection.ts`: Exports `withDatabaseConnection`, a helper that provides a database context to any function.
- Other files: Contain table-specific functions, with each file corresponding to a database table.

## `crawler/ml`

This module handles machine learning tasks, such as content classification.

- `manager.ts`: Defines a base class `AIManager` for managing ML models.
- `akari.ts`: Implements our primary classification model, `AkariProto`, which extends `AIManager`. It filters videos to determine if they should be included as songs.

## `crawler/mq`

This module manages task queuing and processing through BullMQ.

## `crawler/mq/exec`

Contains the functions executed by BullMQ workers. Examples include `getVideoInfoWorker` and `takeBulkSnapshotForVideosWorker`.

> **Terminology note:**  
> In this documentation:
> - Functions in `crawler/mq/exec` are called **workers**.  
> - Functions in `crawler/mq/workers` are called **BullMQ workers**.

**Design detail:**  
Since BullMQ requires one handler per queue, we use a `switch` statement inside each BullMQ worker to route jobs based on their name to the correct function in `crawler/mq/exec`.

## `crawler/mq/workers`

Houses the BullMQ worker functions. Each function handles jobs for a specific queue.

## `crawler/mq/task`

To keep worker functions clean and focused, reusable logic is extracted into this directory as **tasks**. These tasks are then imported and used by the worker functions.

## `crawler/net`

This module handles all data fetching operations. Its core component is the `NetworkDelegate`, defined in `net/delegate.ts`.

## `crawler/net/delegate.ts`

Implements robust network request handling, including:

- Rate limiting by task type and proxy
- Support for serverless functions to dynamically rotate requesting IPs

## `crawler/utils`

A collection of utility functions shared across the crawler modules.

## `crawler/src`

Contains the main entry point of the crawler.

We use [concurrently](https://www.npmjs.com/package/concurrently) to run multiple scripts in parallel, enabling efficient execution of various processes.
