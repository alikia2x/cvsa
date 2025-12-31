export interface Singer {
	name: string;
	color?: string;
	birthday?: string;
	message?: string;
}

export const singers: Singer[] = [
	{
		birthday: "0712",
		color: "#66CCFF",
		name: "洛天依",
	},
	{
		birthday: "0711",
		color: "#00FFCC",
		name: "言和",
	},
	{
		birthday: "0412",
		color: "#EE0000",
		name: "乐正绫",
	},
	{
		birthday: "1002",
		color: "#006666",
		name: "乐正龙牙",
	},
	{
		birthday: "1210",
		color: "#0080FF",
		name: "徵羽摩柯",
	},
	{
		birthday: "0520",
		color: "#FFFF00",
		name: "墨清弦",
	},
	{
		birthday: "0812",
		color: "#9999FF",
		name: "星尘",
	},
	{
		birthday: "1208",
		color: "#613c8a",
		name: "永夜Minus",
	},
	{
		birthday: "0210",
		color: "#EE82EE",
		name: "心华",
	},
	{
		birthday: "0722",
		color: "#3399FF",
		name: "海伊",
	},
	{
		birthday: "0520",
		color: "#8BC0B5",
		name: "苍穹",
	},
	{
		birthday: "1126",
		color: "#FF4004",
		name: "赤羽",
	},
	{
		birthday: "0119",
		color: "#F6BE72",
		name: "诗岸",
	},
	{
		birthday: "0807",
		color: "#2A2859",
		name: "牧心",
	},
	{
		birthday: "0713",
		color: "#FF0099",
		name: "起礼",
	},
	{
		birthday: "0713",
		color: "#99FF00",
		name: "起复",
	},
	{
		birthday: "1110",
		color: "#34CCCC",
		name: "夏语遥",
	},
];

export const specialSingers = [
	{
		message: "你是我最真模样，从来不曾遗忘。",
		name: "雅音宫羽",
	},
	{
		message: "初始之音，响彻未来!",
		name: "初音未来",
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
