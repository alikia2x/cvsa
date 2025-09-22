import { getSingerForBirthday, pickSinger, pickSpecialSinger, Singer } from "@elysia/lib/singers";
import { VERSION } from "@elysia/src";
import { Elysia, t } from "elysia";

const SingerObj = t.Object({
	name: t.String(),
	color: t.Optional(t.String()),
	birthday: t.Optional(t.String()),
	message: t.Optional(t.String())
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
				name: "中 V 档案馆",
				mascot: "知夏",
				quote: "星河知海夏生光"
			},
			status: 200,
			version: VERSION,
			time: Date.now(),
			singer: singer
		};
	},
	{
		response: {
			200: t.Object({
				project: t.Object({
					name: t.String(),
					mascot: t.String(),
					quote: t.String()
				}),
				status: t.Number(),
				version: t.String(),
				time: t.Number(),
				singer: t.Union([SingerObj, t.Array(SingerObj)])
			})
		},
		detail: {
			summary: "Root route",
			description: "The root path. It returns a JSON object containing a random virtual singer, \
			backend version, current server time and other miscellaneous information."
		}
	}
);
