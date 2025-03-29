import { Queue } from "bullmq";

export const LatestVideosQueue = new Queue("latestVideos");

export const ClassifyVideoQueue = new Queue("classifyVideo");

export const SnapshotQueue = new Queue("snapshot");
