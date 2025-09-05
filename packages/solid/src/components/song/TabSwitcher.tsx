import { Button } from "@m3-components/solid";
import { A } from "@solidjs/router";
import { Component, splitProps } from "solid-js";
import { ElementProps } from "../common";

export const TabSwitcher: Component<ElementProps> = (props) => {
	const [_v, rest] = splitProps(props, ["class"]);

	return (
		<nav class="flex flex-col" {...rest}>
			<div class="w-full lg:w-48 gap-4 flex overflow-auto lg:flex-col items-center lg:self-center 2xl:self-end">
				<A class="min-w-20 w-full" href="../info">
					<Button class="w-full" variant="filled">
						信息
					</Button>
				</A>
				<A class="min-w-20 w-full" href="../lyrics">
					<Button class="w-full" variant="outlined">
						歌词
					</Button>
				</A>
				<A class="min-w-20 w-full" href="../analytics">
					<Button class="w-full" variant="outlined">
						数据
					</Button>
				</A>
				<A class="min-w-20 w-full" href="../relations">
					<Button class="w-full" variant="outlined">
						相关
					</Button>
				</A>
				<A class="min-w-20 w-full" href="../discussion">
					<Button class="w-full" variant="outlined">
						讨论
					</Button>
				</A>
			</div>
		</nav>
	);
};
