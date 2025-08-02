import { tv } from "tailwind-variants";
import { navigationExpanded, NavigationMobile, NavigationRegion } from "./Navigation";
import { DivProps } from "../common";
import { Component } from "solid-js";
import { BeforeLeaveEventArgs, useBeforeLeave } from "@solidjs/router";
import { refreshTab } from "~/app";

export const BodyRegion: Component<DivProps> = (props) => {
	const bodyStyle = tv({
		base: "relative",
		variants: {
			open: {
				true: "px-5 lg:left-55 lg:pr-55",
				false: "px-5 lg:left-24 lg:pr-24"
			}
		}
	});
	return (
		<div class={bodyStyle({ open: navigationExpanded() })} {...props}>
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
			<NavigationRegion lang={props.lang} />
			<NavigationMobile lang={props.lang} />
			<BodyRegion>
				{props.children}
			</BodyRegion>
		</div>
	);
};
