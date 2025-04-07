import type { Context } from "hono";
import { createHandlers } from "./utils.ts";
import type { BlankEnv, BlankInput } from "hono/types";
import { number, ValidationError } from "yup";
import { getVideoInfo, getVideoInfoByBV } from "@crawler/net/videoInfo";
import { idSchema } from "./snapshots.ts";
import type { VideoInfoData } from "../crawler/net/bilibili.d.ts";

type ContextType = Context<BlankEnv, "/video/:id/info", BlankInput>;

export const videoInfoHandler = createHandlers(async (c: ContextType) => {
	try {
		const id = await idSchema.validate(c.req.param("id"));
		let videoId: string | number = id as string;
		if (videoId.startsWith("av")) {
			videoId = parseInt(videoId.slice(2));
		} else if (await number().isValid(videoId)) {
			videoId = parseInt(videoId);
		}
		let result: VideoInfoData | number;
		if (typeof videoId === "number") {
			result = await getVideoInfo(videoId, "getVideoInfo");
		} else {
			result = await getVideoInfoByBV(videoId, "getVideoInfo");
		}

		if (typeof result === "number") {
			return c.json({ message: "Error fetching video info", code: result }, 500);
		}

		return c.json(result);
	} catch (e) {
		if (e instanceof ValidationError) {
			return c.json({ message: "Invalid query parameters", errors: e.errors }, 400);
		} else {
			return c.json({ message: "Unhandled error", error: e }, 500);
		}
	}
});
