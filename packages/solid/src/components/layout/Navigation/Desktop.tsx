import { Component } from "solid-js";
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

export const NavigationDesktop: Component = () => {
	return (
		<AppBar class="hidden lg:flex h-20 xl:h-22 2xl:h-24" variant="search">
			<AppBarLeadingElement class="h-full grow shrink basis-0">
				<DynamicImage
					class="lg:block h-full"
					darkSrc="/icons/zh/appbar_desktop_dark.svg"
					lightSrc="/icons/zh/appbar_desktop_light.svg"
				/>
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
	);
};
