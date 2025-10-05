import { getSingerForBirthday, pickSinger, pickSpecialSinger, type Singer } from "lib/const/singers";
import { VERSION } from "src/main";
import { createHandlers } from "src/utils";

export const rootHandler = createHandlers((c) => {
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
	return c.json({
		project: {
			name: "中V档案馆",
			motto: "一起唱吧，心中的歌！"
		},
		status: 200,
		version: VERSION,
		time: Date.now(),
		singer: singer
	});
});
