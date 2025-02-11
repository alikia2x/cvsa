import { Queue } from "bullmq";

export const LatestVideosQueue = new Queue("latestVideos");

export const VideoTagsQueue = new Queue("videoTags");
