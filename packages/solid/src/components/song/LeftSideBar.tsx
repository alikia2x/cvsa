import { Component } from "solid-js";
import { A } from "@solidjs/router";
import { Button } from "@m3-components/solid";
import { HomeIcon, MusicIcon } from "~/components/icons";
import { StarBadge4, StarBadge6, StarBadge8 } from "~/components/icons/StarBadges";
import { HistoryIcon } from "../icons/History";

export const LeftSideBar: Component = () => {
	return (
		<>
			<div class="inline-flex flex-col gap-4">
				<A href="/">
					<Button variant="outlined" class="gap-1 items-center" size="extra-small">
						<HomeIcon class="w-5 h-5 text-xl -translate-y-0.25" />
						<span>主页</span>
					</Button>
				</A>
				<A href="/songs">
					<Button variant="outlined" class="gap-1 items-center" size="extra-small">
						<MusicIcon class="w-5 h-5 text-xl" />
						<span>歌曲</span>
					</Button>
				</A>
				<A href="/milestone/denndou/songs">
					<Button variant="outlined" class="gap-1 items-center" size="extra-small">
						<StarBadge4 class="w-5 h-5 text-xl" />
						<span>殿堂曲</span>
					</Button>
				</A>
				<A href="/milestone/densetsu/songs">
					<Button variant="outlined" class="gap-1 items-center" size="extra-small">
						<StarBadge6 class="w-5 h-5 text-xl" />
						<span>传说曲</span>
					</Button>
				</A>
				<A href="/milestone/shinwa/songs">
					<Button variant="outlined" class="gap-1 items-center" size="extra-small">
						<StarBadge8 class="w-5 h-5 text-xl" />
						<span>神话曲</span>
					</Button>
				</A>
				<A href="../history">
					<Button variant="outlined" class="gap-1 items-center" size="extra-small">
						<HistoryIcon class="w-5 h-5 text-xl" />
						<span>页面历史</span>
					</Button>
				</A>
			</div>
		</>
	);
};
