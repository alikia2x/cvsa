import { createSignal, For } from "solid-js";
import { HomeIcon } from "../icons/Home";
import { MusicIcon } from "../icons/Music";
import {
	NavigationRailFAB,
	NavigationRail,
	NavigationRailAction,
	NavigationRailActions,
	NavigationRailMenu
} from "@m3-components/solid";
import { A } from "@solidjs/router";
import { AlbumIcon } from "~/components/icons/Album";
import { SearchIcon } from "../icons/Search";

export const [activeTab, setActiveTab] = createSignal(-1);
export const [navigationExpanded, setNavigationExpanded] = createSignal(false);
export const actions = [
	{
		icon: <HomeIcon />,
		label: "主页",
		href: "/"
	},
	{
		icon: <MusicIcon />,
		label: "歌曲",
		href: "/songs"
	},
	{
		icon: <AlbumIcon />,
		label: "专辑",
		href: "/albums"
	}
];

export const NavigationRegion = () => {
	return (
		<NavigationRail class="top-0 bg-surface-container" width={220} expanded={navigationExpanded()}>
			<NavigationRailMenu onClick={() => setNavigationExpanded(!navigationExpanded())} />
			<NavigationRailFAB text="搜索" color="primary">
				<SearchIcon />
			</NavigationRailFAB>
			<NavigationRailActions>
				<For each={actions}>
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
