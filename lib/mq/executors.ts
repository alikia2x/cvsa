import { Job } from "bullmq";
import { redis } from "lib/db/redis.ts";
import { insertLatestVideos } from "lib/task/insertLatestVideo.ts";
import MainQueue from "lib/mq/index.ts";
import { MINUTE, SECOND } from "$std/datetime/constants.ts";
import { db } from "lib/db/init.ts";
import { truncate } from "lib/utils/truncate.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";

const LAST_EXECUTED_KEY = "job:insert-videos:last-executed";
const DELTA = 15 * SECOND;
const delayMap = [5, 10, 15, 30, 60, 60];

const setLastExecutedTimestamp = async () => {
    await redis.set(LAST_EXECUTED_KEY, Date.now());
    console.log(`[redis] job:getLatestVideos last executed timestamp set to ${Date.now()}`);
}

const addJobToQueue = async (failedCount: number, delay: number) => {
    const job = await MainQueue.getJob("getLatestVideos");
    if (job && job.getState() === 'active') {
        console.log(`[bullmq] job:getLatestVideos is already running.`);
        return;
    }
    console.log(`[bullmq] job:getLatestVideos added to queue with delay of ${delay / MINUTE} minutes.`)
    MainQueue.add("getLatestVideos", { failedCount }, { delay: delay })
};

export const insertVideosWorker = async (job: Job) => {
    const failedCount = (job.data.failedCount ?? 0) as number;
    const client = await db.connect();
    const lastExecutedTimestamp = Number(await redis.get(LAST_EXECUTED_KEY));
    console.log(`[redis] job:getLatestVideos last executed at ${new Date(lastExecutedTimestamp).toISOString()}`)

    if (!lastExecutedTimestamp || isNaN(lastExecutedTimestamp)) {
        await executeTask(client, failedCount);
        return;
    }

    const diff = Date.now() - lastExecutedTimestamp;
    if (diff < 5 * MINUTE) {
        const waitTime = 5 * MINUTE - diff;
        await addJobToQueue(0, waitTime + DELTA);
        return;
    }

    await executeTask(client, failedCount);
};

const executeTask = async (client: Client, failedCount: number) => {
    console.log("[task] Executing task:getLatestVideos")
    const result = await insertLatestVideos(client);
    failedCount = result !== 0 ? truncate(failedCount + 1, 0, 5) : 0;
    await setLastExecutedTimestamp();
    await addJobToQueue(failedCount, delayMap[failedCount] * MINUTE);
};