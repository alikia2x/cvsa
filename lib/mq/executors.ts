import { Job } from "bullmq";
import { redis } from "lib/db/redis.ts";
import { insertLatestVideos } from "lib/task/insertLatestVideo.ts";
import MainQueue from "lib/mq/index.ts";
import { MINUTE, SECOND } from "$std/datetime/constants.ts";
import { db } from "lib/db/init.ts";
import { truncate } from "lib/utils/turncate.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";

const LAST_EXECUTED_KEY = "job:insert-videos:last-executed";
const DELTA = 15 * SECOND;
const delayMap = [5, 10, 15, 30, 60, 60];

const setLastExecutedTimestamp = async () => 
    await redis.set(LAST_EXECUTED_KEY, Date.now());

const addJobToQueue = (failedCount: number, delay: number) => 
    MainQueue.add("getLatestVideos", { failedCount }, { delay });

export const insertVideosWorker = async (job: Job) => {
    const failedCount = (job.data.failedCount ?? 0) as number;
    const client = await db.connect();
    const lastExecutedTimestamp = Number(await redis.get(LAST_EXECUTED_KEY));

    if (!lastExecutedTimestamp || isNaN(lastExecutedTimestamp)) {
        await executeTask(client, failedCount);
        return;
    }

    const diff = Date.now() - lastExecutedTimestamp;
    if (diff < 5 * MINUTE) {
        addJobToQueue(0, diff + DELTA);
        return;
    }

    await executeTask(client, failedCount);
};

const executeTask = async (client: Client, failedCount: number) => {
    const result = await insertLatestVideos(client);
    await setLastExecutedTimestamp();
    failedCount = result !== 0 ? truncate(failedCount + 1, 0, 5) : 0;
    addJobToQueue(failedCount, delayMap[failedCount] * MINUTE);
};