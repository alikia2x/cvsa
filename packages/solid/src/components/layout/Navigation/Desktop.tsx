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
		<AppBar class="hidden lg:flex h-20 xl:h-22 2xl:h-24 z-20" variant="search">
			<AppBarLeadingElement class="ml-4 h-full grow shrink basis-0">
				<DynamicImage
					class="lg:block h-full"
					darkSrc="/icons/zh/appbar_desktop_dark.svg"
					lightSrc="/icons/zh/appbar_desktop_light.svg"
				/>
			</AppBarLeadingElement>
			<AppBarSearchContainer>
				<AppBarSearchBox
					class="mx-auto text-center placeholder-on-surface-variant text-on-surface 
							placeholder:font-light"
					placeholder="搜索"
				/>
			</AppBarSearchContainer>
			<AppBarTrailingElementGroup class="h-full grow shrink basis-0">
				<AppBarTrailingElement>
					<IconButton></IconButton>
				</AppBarTrailingElement>
			</AppBarTrailingElementGroup>
		</AppBar>
	);
};
