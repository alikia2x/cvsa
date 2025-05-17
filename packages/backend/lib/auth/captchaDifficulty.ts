import { Psql } from "@core/db/psql";
import { SlidingWindow } from "@core/mq/slidingWindow.ts";
import { redis } from "@core/db/redis.ts";
import { getIdentifier } from "@/middleware/rateLimiters.ts";
import { Context } from "hono";

type seconds = number;

export interface CaptchaDifficultyConfig {
	global: boolean;
	duration: seconds;
	threshold: number;
	difficulty: number;
}

export const getCaptchaDifficultyConfigByRoute = async (sql: Psql, route: string): Promise<CaptchaDifficultyConfig[]> =>  {
	return sql<CaptchaDifficultyConfig[]>`
		SELECT duration, threshold, difficulty, global
		FROM captcha_difficulty_settings
		WHERE CONCAT(method, '-', path) = ${route}
		ORDER BY duration
	`;
};

export const getCaptchaConfigMaxDuration = async (sql: Psql, route: string): Promise<seconds> => {
	const rows = await sql<{max: number}[]>`
		SELECT MAX(duration)
		FROM captcha_difficulty_settings
		WHERE CONCAT(method, '-', path) = ${route}
	`;
	if (rows.length < 1){
		return Number.MAX_SAFE_INTEGER;
	}
	return rows[0].max;
}


export const getCurrentCaptchaDifficulty = async (sql: Psql, c: Context | string): Promise<number | null> => {
	const isRoute = typeof c === "string";
	const route = isRoute ? c : `${c.req.method}-${c.req.path}`
	const configs = await getCaptchaDifficultyConfigByRoute(sql, route);
	if (configs.length < 1) {
		return null
	}
	else if (configs.length == 1) {
		return configs[0].difficulty
	}
	const maxDuration = configs.reduce((max, config) =>
		Math.max(max, config.duration), 0);
	const slidingWindow = new SlidingWindow(redis, maxDuration);
	for (let i = 1; i < configs.length; i++) {
		const config = configs[i];
		const lastConfig = configs[i - 1];
		const identifier = isRoute ? c : getIdentifier(c, config.global);
		const count = await slidingWindow.count(`captcha-${identifier}`, config.duration);
		if (count >= config.threshold) {
			continue;
		}
		return lastConfig.difficulty
	}
	return configs[configs.length-1].difficulty;
}
