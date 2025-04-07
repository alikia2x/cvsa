import { getSingerForBirthday, pickSinger, pickSpecialSinger, type Singer } from "./singers.ts";
import { VERSION } from "./main.ts";
import { createHandlers } from "./utils.ts";

export const rootHandler = createHandlers((c) => {
	let singer: Singer | Singer[] | null = null;
	const shouldShowSpecialSinger = Math.random() < 0.016;
	if (getSingerForBirthday().length !== 0) {
		singer = getSingerForBirthday();
		for (const s of singer) {
			delete s.birthday;
			s.message = `祝${s.name}生日快乐~`;
		}
	} else if (shouldShowSpecialSinger) {
		singer = pickSpecialSinger();
	} else {
		singer = pickSinger();
	}
	return c.json({
		"project": {
			"name": "中V档案馆",
			"motto": "一起唱吧，心中的歌！",
		},
		"status": 200,
		"version": VERSION,
		"time": Date.now(),
		"singer": singer,
	});
});
