import { Component, createSignal } from "solid-js";
import { AlbumIcon, HomeIcon, MusicIcon } from "~/components/icons";

export const [activeTab, setActiveTab] = createSignal(-1);
export const [navigationExpanded, setNavigationExpanded] = createSignal(false);

interface Action {
	icon: Component;
	label: string;
	href: string;
}

export const actions: Action[] = [
	{
		icon: HomeIcon,
		label: "主页",
		href: "/"
	},
	{
		icon: MusicIcon,
		label: "歌曲",
		href: "/songs"
	},
	{
		icon: AlbumIcon,
		label: "专辑",
		href: "/albums"
	}
];

export const actionsEn: Action[] = [
	{
		icon: HomeIcon,
		label: "Home",
		href: "/en/"
	},
	{
		icon: MusicIcon,
		label: "Songs",
		href: "/en/songs"
	},
	{
		icon: AlbumIcon,
		label: "Albums",
		href: "/en/albums"
	}
];

export const tabMap = {
	"/": 0,
	"/song*": 1,
	"/song/**/*": 1,
	"/albums": 2,
	"/album/**/*": 2,
	"/en/": 0,
	"/en/songs": 1,
	"/en/song*": 1,
	"/en/song/**/*": 1,
	"/en/albums": 2,
	"/en/album/**/*": 2
};

export const searchT = {
	zh: "搜索",
	en: "Search"
};
