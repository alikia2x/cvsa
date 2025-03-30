import express from "express";
import { createBullBoard } from "@bull-board/api";
import { BullMQAdapter } from "@bull-board/api/bullMQAdapter.js";
import { ExpressAdapter } from "@bull-board/express";
import { ClassifyVideoQueue, LatestVideosQueue, SnapshotQueue } from "mq/index.ts";

const serverAdapter = new ExpressAdapter();
serverAdapter.setBasePath("/");

createBullBoard({
	queues: [
		new BullMQAdapter(LatestVideosQueue),
		new BullMQAdapter(ClassifyVideoQueue),
		new BullMQAdapter(SnapshotQueue),
	],
	serverAdapter: serverAdapter,
});

const app = express();

app.use("/", serverAdapter.getRouter());

app.listen(3000, () => {
	console.log("Running on 3000...");
	console.log("For the UI, open http://localhost:3000/");
	console.log("Make sure Redis is running on port 6379 by default");
});
