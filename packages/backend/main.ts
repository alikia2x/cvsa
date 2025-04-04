import { Hono } from "hono";
import { dbCredMiddleware, dbMiddleware } from "./database.ts";
import { rootHandler } from "./root.ts";
import { getSnapshotsHanlder } from "./snapshots.ts";
import { registerHandler } from "./register.ts";

export const app = new Hono();

app.use('/video/*', dbMiddleware);
app.use('/user', dbCredMiddleware);

app.get("/", ...rootHandler);

app.get('/video/:id/snapshots', ...getSnapshotsHanlder);
app.post('/user', ...registerHandler);

const fetch = app.fetch;

export default {
	fetch,
} satisfies Deno.ServeDefaultExport;

export const VERSION = "0.3.0";