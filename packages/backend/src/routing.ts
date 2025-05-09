import { rootHandler } from "routes";
import { pingHandler } from "routes/ping";
import { registerHandler } from "routes/user";
import { videoInfoHandler, getSnapshotsHanlder } from "routes/video";
import { Hono } from "hono";
import { Variables } from "hono/types";

export function configureRoutes(app: Hono<{ Variables: Variables }>) {
	app.get("/", ...rootHandler);
	app.all("/ping", ...pingHandler);

	app.get("/video/:id/snapshots", ...getSnapshotsHanlder);
	app.post("/user", ...registerHandler);

	app.get("/video/:id/info", ...videoInfoHandler);
}
