import { assertEquals } from "jsr:@std/assert";
import { getLatestVideos } from "lib/net/getLatestVideos.ts";

Deno.test("Get latest videos", async () => {
	const videos = (await getLatestVideos(1, 5))!;
	assertEquals(videos.length, 5);

	videos.forEach((video) => {
		assertVideoProperties(video);
	});
});

function assertVideoProperties(video: object) {
	const aid = "aid" in video && typeof video.aid === "number";
	const bvid = "bvid" in video && typeof video.bvid === "string" &&
		video.bvid.length === 12 && video.bvid.startsWith("BV");
	const description = "description" in video && typeof video.description === "string";
	const uid = "uid" in video && typeof video.uid === "number";
	const tags = "tags" in video && (typeof video.tags === "string" || video.tags === null);
	const title = "title" in video && typeof video.title === "string";
	const publishedAt = "published_at" in video && typeof video.published_at === "string";

	const match = aid && bvid && description && uid && tags && title && publishedAt;
	assertEquals(match, true);
}
