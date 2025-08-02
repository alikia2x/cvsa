import { Component, createEffect, createMemo, createSignal, For, on, onMount } from "solid-js";
import { HomeIcon } from "../icons/Home";
import { MusicIcon } from "../icons/Music";
import {
	NavigationRailFAB,
	NavigationRail,
	NavigationRailAction,
	NavigationRailActions,
	NavigationRailMenu,
	AppBar,
	AppBarLeadingElement,
	AppBarSearchBox,
	AppBarTrailingElementGroup,
	AppBarTrailingElement,
	IconButton
} from "@m3-components/solid";
import { A } from "@solidjs/router";
import { AlbumIcon } from "~/components/icons/Album";
import { SearchIcon } from "../icons/Search";
import { Portal } from "solid-js/web";
import { animate, createTimer, utils } from "animejs";
import { tv } from "tailwind-variants";

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

const searchT = {
	zh: "搜索",
	en: "Search"
};

export const NavigationMobile: Component<{ lang?: "zh" | "en" }> = (props) => {
	const [el, setEl] = createSignal<HTMLElement | null>(null);

	createEffect(() => {
		if (!el) return;
		if (navigationExpanded()) {
			animate(el()!, {
				x: 0,
				duration: 500,
				z: 100,
				ease: "cubicBezier(0.27, 1.06, 0.18, 1.00)"
			});
		} else {
			animate(el()!, {
				x: -340,
				duration: 500,
				z: 0,
				ease: "cubicBezier(0.27, 1.06, 0.18, 1.00)"
			});
		}
	});

	return (
		<>
			<NavigationRailMenu
				class="top-3 left-4 fixed z-100 bg-surface-container/60 backdrop-blur-md lg:hidden"
				onClick={() => {
					setNavigationExpanded(!navigationExpanded());
				}}
			/>
			<AppBar class="lg:hidden" variant="search">
				<AppBarLeadingElement>
					<NavigationRailMenu class="invisible" />
				</AppBarLeadingElement>
				<AppBarSearchBox placeholder="搜索" />
				<AppBarTrailingElementGroup>
					<AppBarTrailingElement>
						<IconButton></IconButton>
					</AppBarTrailingElement>
				</AppBarTrailingElementGroup>
			</AppBar>
			<Portal mount={document.getElementById("modal") || undefined}>
				<div
					class="fixed lg:invisible top-0 left-0 h-full z-50"
					style="transform: translateX(-300px);"
					ref={(el) => {
						setEl(el);
					}}
				>
					<NavigationRail
						class="top-0 bg-surface-container rounded-r-2xl shadow-shadow shadow-2xl"
						width={256}
						expanded={true}
					>
						<NavigationRailFAB text={searchT[props.lang || "zh"]} class="pr-6 mt-6" color="primary">
							<SearchIcon />
						</NavigationRailFAB>
						<NavigationRailActions>
							<For each={props.lang == "en" ? actionsEn : actions}>
								{(action, index) => (
									<A href={action.href} class="clear">
										<NavigationRailAction
											activated={activeTab() == index()}
											label={action.label}
											icon={action.icon}
											onClick={() => {
												setActiveTab(index);
											}}
										/>
									</A>
								)}
							</For>
						</NavigationRailActions>
					</NavigationRail>
				</div>
			</Portal>
		</>
	);
};

export const NavigationRegion: Component<{ lang?: "zh" | "en" }> = (props) => {
	return (
		<NavigationRail class="hidden lg:flex top-0 bg-surface-container" width={220} expanded={navigationExpanded()}>
			<NavigationRailMenu class="lg:flex left-7" onClick={() => setNavigationExpanded(!navigationExpanded())} />
			<NavigationRailFAB text={searchT[props.lang || "zh"]} color="primary">
				<SearchIcon />
			</NavigationRailFAB>
			<NavigationRailActions>
				<For each={props.lang == "en" ? actionsEn : actions}>
					{(action, index) => (
						<A href={action.href} class="clear">
							<NavigationRailAction
								activated={activeTab() == index()}
								label={action.label}
								icon={action.icon}
								onClick={() => {
									setActiveTab(index);
								}}
							/>
						</A>
					)}
				</For>
			</NavigationRailActions>
		</NavigationRail>
	);
};
