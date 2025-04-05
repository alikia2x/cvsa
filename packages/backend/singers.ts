export const singers = [
	{
		"name": "洛天依",
		"color": "#66CCFF",
		"birthday": "0712",
	},
	{
		"name": "言和",
		"color": "#00FFCC",
		"birthday": "0711",
	},
	{
		"name": "乐正绫",
		"color": "#EE0000",
		"birthday": "0412",
	},
	{
		"name": "乐正龙牙",
		"color": "#006666",
		"birthday": "1002",
	},
	{
		"name": "徵羽摩柯",
		"color": "#0080FF",
		"birthday": "1210",
	},
	{
		"name": "墨清弦",
		"color": "#FFFF00",
		"birthday": "0520",
	},
	{
		"name": "星尘",
		"color": "#9999FF",
		"birthday": "0812",
	},
	{
		"name": "心华",
		"color": "#EE82EE",
		"birthday": "0210",
	},
	{
		"name": "海伊",
		"color": "#3399FF",
		"birthday": "0722",
	},
	{
		"name": "苍穹",
		"color": "#8BC0B5",
		"birthday": "0520",
	},
	{
		"name": "赤羽",
		"color": "#FF4004",
		"birthday": "1126",
	},
	{
		"name": "诗岸",
		"color": "#F6BE72",
		"birthday": "0119",
	},
	{
		"name": "牧心",
		"color": "#2A2859",
		"birthday": "0807",
	},
];

export interface Singer {
	name: string;
	color?: string;
	birthday?: string;
	message?: string;
}

export const specialSingers = [
	{
		"name": "雅音宫羽",
		"message": "你是我最真模样，从来不曾遗忘。",
	},
	{
		"name": "初音未来",
		"message": "初始之音，响彻未来!",
	},
];

export const pickSinger = () => {
	const index = Math.floor(Math.random() * singers.length);
	return singers[index];
};

export const pickSpecialSinger = () => {
	const index = Math.floor(Math.random() * specialSingers.length);
	return specialSingers[index];
};

export const getSingerForBirthday = (): Singer[] => {
	const today = new Date();
	const month = String(today.getMonth() + 1).padStart(2, "0");
	const day = String(today.getDate()).padStart(2, "0");
	const datestring = `${month}${day}`;
	return singers.filter((singer) => singer.birthday === datestring);
};
