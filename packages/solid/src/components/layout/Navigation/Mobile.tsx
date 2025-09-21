import { Component, createEffect, createSignal, For } from "solid-js";
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
	IconButton,
	AppBarSearchContainer,
	ExtendedFAB
} from "@m3-components/solid";
import { A } from "@solidjs/router";
import { SearchIcon } from "~/components/icons/Search";
import { Portal } from "solid-js/web";
import { animate } from "animejs";
import { actions, actionsEn, activeTab, navigationExpanded, searchT, setActiveTab, setNavigationExpanded } from ".";

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
				x: -380,
				duration: 500,
				z: 0,
				ease: "cubicBezier(0.27, 1.06, 0.18, 1.00)"
			});
		}
	});

	return (
		<>
			<NavigationRailMenu
				class="top-3 left-4 fixed z-[100] backdrop-blur-md shadow-xl lg:hidden"
				onClick={() => {
					setNavigationExpanded(!navigationExpanded());
				}}
			/>
			<AppBar class="z-20 lg:hidden" variant="search">
				<AppBarLeadingElement>
					<NavigationRailMenu class="invisible" />
				</AppBarLeadingElement>
				<AppBarSearchContainer class="max-sm:w-[calc(100%-7.9rem)]">
					<AppBarSearchBox placeholder="搜索" class="placeholder-on-surface-variant text-on-surface" />
				</AppBarSearchContainer>
				<AppBarTrailingElementGroup>
					<AppBarTrailingElement>
						<IconButton></IconButton>
					</AppBarTrailingElement>
				</AppBarTrailingElementGroup>
			</AppBar>
			<Portal mount={document.getElementById("modal") || undefined}>
				<div
					class="fixed lg:hidden top-0 left-0 h-full z-50"
					style="transform: translateX(-300px);"
					ref={(el) => {
						setEl(el);
					}}
				>
					<NavigationRail
						class="z-20 top-0 bg-background overflow-auto rounded-r-2xl shadow-shadow shadow-2xl"
						width={256}
						expanded={true}
					>
						<ExtendedFAB
							text={searchT[props.lang || "zh"]}
							class="left-5 top-5 font-medium leading-6 duration-100 whitespace-nowrap
							transition-none w-24 h-14 text-base mt-6"
							color="primary"
							position="unset"
						>
							<SearchIcon />
						</ExtendedFAB>
						<NavigationRailActions>
							<For each={props.lang == "en" ? actionsEn : actions}>
								{(action, index) => (
									<A href={action.href} class="clear">
										<NavigationRailAction
											activated={activeTab() == index()}
											label={action.label}
											icon={action.icon}
											onClick={() => {
												setNavigationExpanded(false);
												setActiveTab(index);
											}}
										/>
									</A>
								)}
							</For>
						</NavigationRailActions>
					</NavigationRail>
					<div
						onclick={() => setNavigationExpanded(false)}
						class="w-screen h-screen z-10 absoluter bg-transparent"
					></div>
				</div>
			</Portal>
		</>
	);
};
