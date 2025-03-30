import { Hono } from "hono";
import { dbMiddleware } from "./database.ts";
import { rootHandler } from "./root.ts";
import { getSnapshotsHanlder } from "./snapshots.ts";

export const app = new Hono();

app.use('/video/*', dbMiddleware);

app.get("/", ...rootHandler);

app.get('/video/:id/snapshots', ...getSnapshotsHanlder);

const fetch = app.fetch;

export default {
	fetch,
} satisfies Deno.ServeDefaultExport;

export const VERSION = "0.2.4";