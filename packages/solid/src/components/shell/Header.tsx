import { Component } from "solid-js";
import {
	AppBar,
	AppBarSearchBox,
	IconButton,
	LeadingElement,
	TrailingElementGroup,
	TrailingElement
} from "@m3-components/solid";
import "@m3-components/solid/index.css";
import { MenuOpen } from "~/components/icons/MenuOpen";

export const Header: Component = () => {
	return (
		<div class="mt-4 top-0 left-0 w-full">
			<AppBar variant="search">
				<LeadingElement>
					<IconButton>
						<MenuOpen/>
					</IconButton>
				</LeadingElement>
				<AppBarSearchBox class="text-center placeholder:text-on-surface-variant text-on-surface" placeholder="搜索" />
				<TrailingElementGroup>
					<TrailingElement>
						<IconButton></IconButton>
					</TrailingElement>
				</TrailingElementGroup>
			</AppBar>
		</div>
	);
};
