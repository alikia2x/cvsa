import { NavigationMobile } from "./Navigation";
import { DivProps } from "../common";
import { Component } from "solid-js";
import { BeforeLeaveEventArgs, useBeforeLeave } from "@solidjs/router";
import { refreshTab } from "~/app";
import LogoLight from "/icons/zh/appbar_desktop_light.svg";
import LogoDark from "/icons/zh/appbar_desktop_dark.svg";
import { DynamicImage } from "~/components/utils/DynamicImage";
import {
	AppBar,
	AppBarLeadingElement,
	AppBarSearchBox,
	AppBarSearchContainer,
	AppBarTrailingElement,
	AppBarTrailingElementGroup,
	IconButton
} from "@m3-components/solid";

export const BodyRegion: Component<DivProps> = (props) => {
	return (
		<div class="pt-12 px-4" {...props}>
			{props.children}
		</div>
	);
};

interface LayoutProps extends DivProps {
	lang?: "zh" | "en";
}

export const Layout: Component<LayoutProps> = (props) => {
	useBeforeLeave((e: BeforeLeaveEventArgs) => {
		if (typeof e.to === "number") {
			refreshTab(e.to.toString());
			return;
		}
		refreshTab(e.to);
	});
	return (
		<div class="relatve w-screen min-h-screen">
			<NavigationMobile lang={props.lang} />
			<AppBar class="hidden lg:flex h-20 xl:h-22 2xl:h-24" variant="search">
				<AppBarLeadingElement class="h-full grow shrink basis-0">
					<DynamicImage class="lg:block h-full" darkSrc={LogoDark} lightSrc={LogoLight} />
				</AppBarLeadingElement>
				<AppBarSearchContainer>
					<AppBarSearchBox class="mx-auto text-center" placeholder="搜索" />
				</AppBarSearchContainer>
				<AppBarTrailingElementGroup class="h-full grow shrink basis-0">
					<AppBarTrailingElement>
						<IconButton></IconButton>
					</AppBarTrailingElement>
				</AppBarTrailingElementGroup>
			</AppBar>
			<BodyRegion>{props.children}</BodyRegion>
		</div>
	);
};
