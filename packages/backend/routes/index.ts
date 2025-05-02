import { rootHandler } from "./root/root.ts";
import { pingHandler } from "./ping.ts";
import { getSnapshotsHanlder } from "./snapshots.ts";
import { registerHandler } from "./user.ts";
import { videoInfoHandler } from "db/videoInfo.ts";
import { Hono } from "hono";
import { Variables } from "hono/types";

export function configureRoutes(app: Hono<{Variables: Variables }>) {
	app.get("/", ...rootHandler);
	app.all("/ping", ...pingHandler);

	app.get("/video/:id/snapshots", ...getSnapshotsHanlder);
	app.post("/user", ...registerHandler);

	app.get("/video/:id/info", ...videoInfoHandler);
}