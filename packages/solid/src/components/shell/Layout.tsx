import { tv } from "tailwind-variants";
import { navigationExpanded, NavigationRegion } from "./Navigation";
import { DivProps } from "../common";
import { Component } from "solid-js";
import { BeforeLeaveEventArgs, useBeforeLeave } from "@solidjs/router";
import { refreshTab } from "~/app";

export const BodyRegion: Component<DivProps> = (props) => {
	const bodyStyle = tv({
		base: "relative",
		variants: {
			open: {
				true: "left-55 pr-55",
				false: "left-24 pr-24"
			}
		}
	});
	return (
		<div class={bodyStyle({ open: navigationExpanded() })} {...props}>
			{" "}
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
				<main class="w-full px-6 lg:max-w-4xl lg:mx-auto pt-20">
					{props.children}
				</main>
			</BodyRegion>
		</div>
	);
};
