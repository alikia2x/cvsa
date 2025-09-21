import { NavigationMobile } from "./Navigation/Mobile";
import { DivProps } from "../common";
import { Component } from "solid-js";
import { BeforeLeaveEventArgs, useBeforeLeave } from "@solidjs/router";
import { refreshTab } from "~/app";
import { NavigationDesktop } from "./Navigation/Desktop";
import { BodyRegion } from "./Body";

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
		<div class="relatve w-screen max-w-full min-h-screen">
			<NavigationMobile lang={props.lang} />
			<NavigationDesktop />
			<BodyRegion>{props.children}</BodyRegion>
		</div>
	);
};
