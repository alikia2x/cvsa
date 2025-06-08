import { rootHandler } from "routes";
import { pingHandler } from "routes/ping";
import { getUserByLoginSessionHandler, registerHandler } from "routes/user";
import { videoInfoHandler, getSnapshotsHanlder } from "routes/video";
import { Hono } from "hono";
import { Variables } from "hono/types";
import { createCaptchaSessionHandler, verifyChallengeHandler } from "routes/captcha";
import { getCaptchaDifficultyHandler } from "routes/captcha/difficulty/GET.ts";
import { getVideosHanlder } from "@/routes/videos";
import { loginHandler } from "@/routes/login/session/POST";
import { logoutHandler } from "@/routes/session";

export function configureRoutes(app: Hono<{ Variables: Variables }>) {
	app.get("/", ...rootHandler);
	app.all("/ping", ...pingHandler);

	app.get("/videos", ...getVideosHanlder);

	app.get("/video/:id/snapshots", ...getSnapshotsHanlder);
	app.get("/video/:id/info", ...videoInfoHandler);

	app.post("/login/session", ...loginHandler);

	app.delete("/session/:id", ...logoutHandler);

	app.post("/user", ...registerHandler);
	app.get("/user/session/:id", ...getUserByLoginSessionHandler);

	app.post("/captcha/session", ...createCaptchaSessionHandler);
	app.get("/captcha/:id/result", ...verifyChallengeHandler);
	app.get("/captcha/difficulty", ...getCaptchaDifficultyHandler);
}
