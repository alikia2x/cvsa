import { Component } from "solid-js";
import { AppBar, AppBarSearchBox, IconButton, TrailingElementGroup, TrailingElement } from "@m3-components/solid";
import "@m3-components/solid/index.css";

export const Header: Component = () => {
	return (
		<AppBar variant="search" class="mt-4">
			<AppBarSearchBox
				class="text-center placeholder:text-on-surface-variant text-on-surface"
				placeholder="搜索"
			/>
		</AppBar>
	);
};
