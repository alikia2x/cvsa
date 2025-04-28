import { Hono } from "hono";
import { rootHandler } from "./root.ts";
import { getSnapshotsHanlder } from "./snapshots.ts";
import { registerHandler } from "./register.ts";
import { videoInfoHandler } from "./videoInfo.ts";
import { pingHandler } from "./ping.ts";
// import { getConnInfo } from "hono/deno";
//import { rateLimiter } from "hono-rate-limiter";
// import { MINUTE } from "https://deno.land/std@0.216.0/datetime/constants.ts";
// import type { Context } from "hono";
// import type { BlankEnv } from "hono/types";

export const app = new Hono();

// const limiter = rateLimiter<BlankEnv, "/user", {}>({
// 	windowMs: 60 * MINUTE,
// 	limit: 5,
// 	standardHeaders: "draft-6",
// 	keyGenerator: (c) => {
// 		const info = getConnInfo(c as unknown as Context<BlankEnv, "/user", {}>);
// 		if (!info.remote || !info.remote.address) {
// 			return crypto.randomUUID()
// 		}
// 		return info.remote.address;
// 	},
// });
// app.use("/user", limiter);

app.get("/", ...rootHandler);
app.get("/ping", ...pingHandler);

app.get("/video/:id/snapshots", ...getSnapshotsHanlder);
app.post("/user", ...registerHandler);

app.get("/video/:id/info", ...videoInfoHandler);

export default app

export const VERSION = "0.4.2";
