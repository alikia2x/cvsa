import { tv } from "tailwind-variants";
import { navigationExpanded, NavigationRegion } from "./Navigation";
import { DivProps } from "../common";
import { Component } from "solid-js";

export const BodyRegion: Component<DivProps> = (props) => {
	const bodyStyle = tv({
		base: "relative",
		variants: {
			open: {
				true: "left-90",
				false: "left-24"
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
	return (
		<div class="relatve w-screen min-h-screen">
			<NavigationRegion />
			<BodyRegion>{props.children}</BodyRegion>
		</div>
	);
};
