import { tv } from "tailwind-variants";
import { navigationExpanded, NavigationRegion } from "./Navigation";
import { DivProps } from "../common";
import { Component } from "solid-js";
import { BeforeLeaveEventArgs, useBeforeLeave } from "@solidjs/router";
import { refreshTab } from "~/app";

export const BodyRegion: Component<DivProps> = (props) => {
	const bodyStyle = tv({
		base: "relative px-6 pt-20",
		variants: {
			open: {
				true: "left-55 pr-55",
				false: "left-24 pr-24"
			}
		}
	});
	return (
		<div class={bodyStyle({ open: navigationExpanded() })} {...props}>
			{props.children}
		</div>
	);
};

export const Layout: Component<DivProps> = (props) => {
	useBeforeLeave((e: BeforeLeaveEventArgs) => {
		if (typeof e.to === "number") {
			refreshTab(e.to.toString());
			return;
		}
		refreshTab(e.to);
	});
	return (
		<div class="relatve w-screen min-h-screen">
			<NavigationRegion />
			<BodyRegion>
				{props.children}
			</BodyRegion>
		</div>
	);
};
