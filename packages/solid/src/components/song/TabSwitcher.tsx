import { Button } from "@m3-components/solid";
import { A } from "@solidjs/router";
import { tv } from "tailwind-variants";
import { navigationExpanded } from "~/components/shell/Navigation";

export const TabSwitcher = () => {
	const tabsContainerStyle = tv({
		base: "w-full lg:w-48 gap-4 flex lg:flex-col items-center",
		variants: {
			expanded: {
				true: "lg:self-start xl:self-center",
				false: "self-center"
			}
		}
	});
	return (
		<nav class="flex flex-col lg:h-screen lg:px-6 lg:pt-12">
			<div class={tabsContainerStyle({ expanded: navigationExpanded() })}>
				<A class="w-full" href="../info">
					<Button class="w-full" variant="filled">
						信息
					</Button>
				</A>
				<A class="w-full" href="../lyrics">
					<Button class="w-full" variant="outlined">
						歌词
					</Button>
				</A>
				<A class="w-full" href="../analytics">
					<Button class="w-full" variant="outlined">
						数据
					</Button>
				</A>
				<A class="w-full" href="../relations">
					<Button class="w-full" variant="outlined">
						相关
					</Button>
				</A>
			</div>
		</nav>
	);
};
