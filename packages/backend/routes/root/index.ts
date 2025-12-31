import {
	getSingerForBirthday,
	pickSinger,
	pickSpecialSinger,
	type Singer,
} from "@backend/lib/singers";
import { VERSION } from "@backend/src";
import { Elysia, t } from "elysia";

const SingerObj = t.Object({
	birthday: t.Optional(t.String()),
	color: t.Optional(t.String()),
	message: t.Optional(t.String()),
	name: t.String(),
});

export const rootHandler = new Elysia().get(
	"/",
	async () => {
		let singer: Singer | Singer[];
		const shouldShowSpecialSinger = Math.random() < 0.016;
		if (getSingerForBirthday().length !== 0) {
			singer = JSON.parse(JSON.stringify(getSingerForBirthday())) as Singer[];
			for (const s of singer) {
				delete s.birthday;
				s.message = `祝${s.name}生日快乐~`;
			}
		} else if (shouldShowSpecialSinger) {
			singer = pickSpecialSinger();
		} else {
			singer = pickSinger();
		}
		return {
			project: {
				mascot: "知夏",
				name: "中V档案馆",
				quote: "星河知海夏生光",
			},
			singer: singer,
			status: 200,
			time: Date.now(),
			version: VERSION,
		};
	},
	{
		detail: {
			description:
				"The root path. It returns a JSON object containing a random virtual singer, \
			backend version, current server time and other miscellaneous information.",
			summary: "Root route",
		},
		response: {
			200: t.Object({
				project: t.Object({
					mascot: t.String(),
					name: t.String(),
					quote: t.String(),
				}),
				singer: t.Union([SingerObj, t.Array(SingerObj)]),
				status: t.Number(),
				time: t.Number(),
				version: t.String(),
			}),
		},
	}
);
