import { Component } from "solid-js";
import { DivProps } from "~/components/common";

export const BodyRegion: Component<DivProps> = (props) => {
	return (
		<div class="w-full min-h-full" {...props}>
			{props.children}
		</div>
	);
};
