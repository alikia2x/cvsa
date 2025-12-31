import { db, history } from "@core/drizzle";
import { type ConnectionOptions, QueueEvents, type QueueEventsListener } from "bullmq";
import { redis } from "bun";

interface CustomListener extends QueueEventsListener {
	addSong: (args: { uid: string; songID: number }, id: string) => void;
}
const queueEvents = new QueueEvents("latestVideos", {
	connection: redis as ConnectionOptions,
});
queueEvents.on<CustomListener>(
	"addSong",
	async ({ uid, songID }: { uid: string; songID: number }) => {
		await db.insert(history).values({
			changedBy: uid,
			changeType: "add-song",
			data: null,
			objectId: songID,
		});
	}
);
