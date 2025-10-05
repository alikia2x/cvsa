import { Component } from "solid-js";
import { A } from "@solidjs/router";
import { IconButton, Typography } from "@m3-components/solid";
import { RightArrow } from "~/components/icons/Arrow";

export const Staff: Component<{ name: string; role: string; num: number }> = (props) => {
	return (
		<A
			href={`/author/${props.name}/info`}
			class="group rounded-[1.25rem] hover:bg-surface-container h-16 flex items-center
								px-4 justify-between"
		>
			<div class="ml-2 flex gap-5 lg:gap-4 grow w-full">
				<span
					class="font-[IPSD] font-medium text-[2rem] text-on-surface-variant"
					style="
						-webkit-text-stroke: var(--md-sys-color-on-surface-variant);
						-webkit-text-stroke-width: 1.2px;
						-webkit-text-fill-color: transparent;"
				>
					{props.num}
				</span>
				<div class="flex flex-col gap-[3px]">
					<Typography.Body variant="large" class="text-on-surface font-medium">
						{props.name}
					</Typography.Body>
					<Typography.Label variant="large" class="text-on-surface-variant">
						{props.role}
					</Typography.Label>
				</div>
			</div>
			<IconButton class="text-on-surface-variant opacity-0 group-hover:opacity-80 duration-200">
				<RightArrow />
			</IconButton>
		</A>
	);
};
