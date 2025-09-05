import { Layout } from "~/components/layout";
import { Button, Card, CardContent, CardMedia, ExtendedFAB, IconButton, Typography } from "@m3-components/solid";

import { TabSwitcher } from "~/components/song/TabSwitcher";
import { EditIcon, HomeIcon, MusicIcon } from "~/components/icons";
import { A } from "@solidjs/router";
import { RightArrow } from "~/components/icons/Arrow";
import { Component } from "solid-js";
import { LinkIcon } from "~/components/icons/Link";
import { StarBadge4, StarBadge6, StarBadge8 } from "~/components/icons/StarBadges";
import { HistoryIcon } from "~/components/icons/History";

const Staff: Component<{ name: string; role: string; num: number }> = (props) => {
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

const Content: Component = () => {
	return (
		<>
			<Card variant="outlined" class="w-full max-lg:rounded-none max-lg:border-none">
				<div class="relative w-full overflow-hidden ">
					<CardMedia
						round={false}
						src="https://i0.hdslb.com/bfs/archive/8ad220336f96e4d2ea05baada3bc04592d56b2a5.jpg"
						referrerpolicy="no-referrer"
						class="relative w-full z-[2]"
					/>
					<div class="h-10 lg:h-0" />
					<CardMedia
						round={false}
						src="https://i0.hdslb.com/bfs/archive/8ad220336f96e4d2ea05baada3bc04592d56b2a5.jpg"
						referrerpolicy="no-referrer"
						class="w-full absolute lg:hidden top-10 z-[1]"
					/>
					<span
						class="left-3 absolute bottom-14 z-10 text-sm text-white/95"
						style="text-shadow:0px 1px 1px rgb(0 0 0 / 0.2) "
					>
						赤羽 & 洛凛
					</span>
					<span
						class="left-3 absolute bottom-3 z-10 font-medium text-4xl text-white/90 "
						style="text-shadow: 0px 1px 0px rgb(0 0 0 / 0.075), 0px 1px 1px rgb(0 0 0 / 0.075), 0px 2px 2px rgb(0 0 0 / 0.075)"
					>
						尘海绘仙缘
					</span>
					<span
						class="font-[Inter] right-3 absolute bottom-10 z-10 text-xl text-white/95"
						style="text-shadow: 0px 1px 2px rgb(0 0 0 / 0.1), 0px 3px 2px rgb(0 0 0 / 0.1), 0px 4px 8px rgb(0 0 0 / 0.1)"
					>
						4:54
					</span>
					<span
						class="font-[Inter] right-3 absolute bottom-3 z-10 text-xl text-white/95"
						style="text-shadow: 0px 1px 2px rgb(0 0 0 / 0.1), 0px 3px 2px rgb(0 0 0 / 0.1), 0px 4px 8px rgb(0 0 0 / 0.1)"
					>
						12,422
						<span class="ml-1 text-sm">再生</span>
					</span>
					<div class="lg:hidden w-full gradient-blur !absolute !h-32">
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
					</div>
				</div>

				<CardContent class="max-lg:hidden px-7 py-6 flex flex-col gap-4">
					<Typography.Display class="leading-[2.75rem]" variant="small">
						尘海绘仙缘
					</Typography.Display>
					<div class="grid grid-cols-2 grid-rows-3 gap-2">
						<div class="flex flex-col">
							<Typography.Label class="text-on-surface-variant" variant="large">
								演唱
							</Typography.Label>
							<Typography.Body variant="large">
								<a href="#">赤羽</a>
							</Typography.Body>
						</div>
						<div class="flex flex-col">
							<Typography.Label class="text-on-surface-variant" variant="large">
								时长
							</Typography.Label>
							<Typography.Body variant="large">4:28</Typography.Body>
						</div>
						<div class="flex flex-col">
							<Typography.Label class="text-on-surface-variant" variant="large">
								投稿
							</Typography.Label>
							<Typography.Body variant="large">
								<a href="#">洛凛</a>
							</Typography.Body>
						</div>
						<div class="flex flex-col">
							<Typography.Label class="text-on-surface-variant" variant="large">
								链接
							</Typography.Label>
							<Typography.Body class="flex gap-2" variant="large">
								<a href="https://www.bilibili.com/video/BV1eaq9Y3EVV/">哔哩哔哩</a>
								<a href="https://vocadb.net/S/742394">VocaDB</a>
							</Typography.Body>
						</div>
						<div class="flex flex-col">
							<Typography.Label class="text-on-surface-variant" variant="large">
								发布时间
							</Typography.Label>
							<Typography.Body variant="large">2024-12-15 12:15:00</Typography.Body>
						</div>
						<div class="flex flex-col">
							<Typography.Label class="text-on-surface-variant" variant="large">
								再生
							</Typography.Label>
							<Typography.Body variant="large">1.24 万 (12,422)</Typography.Body>
						</div>
					</div>
				</CardContent>
			</Card>
			<div class="mx-1 my-6 lg:hidden">
				<TabSwitcher />
			</div>
			<article class="mt-6">
				<Typography.Headline class="mx-4" variant="medium">简介</Typography.Headline>
				<Typography.Body class="mx-4 mt-2" variant="large">
					<span class="font-medium">《尘海绘仙缘》</span>是<a href="#">洛凛</a>于
					<span>
						&VeryThinSpace;2024&VeryThinSpace;年&VeryThinSpace;12&VeryThinSpace;月&VeryThinSpace;15&VeryThinSpace;日
					</span>
					投稿至
					<a href="#">哔哩哔哩</a>的&ThinSpace;<a href="#">Synthesizer V</a>&ThinSpace;
					<span>中文</span>
					<span>原创歌曲</span>, 由<a href="#">赤羽</a>演唱。
				</Typography.Body>
				<div class="h-7" />
				<Typography.Headline class="mx-4" variant="medium">制作人员</Typography.Headline>
				<div class="mt-3 mx-1">
					<Staff num={1} name="洛凛" role="策划、作词" />
					<Staff num={2} name="鱼柳" role="作曲、编曲" />
					<Staff num={3} name="月华" role="混音" />
					<Staff num={4} name="城西阿灵" role="视频" />
					<Staff num={5} name="与嬴酌棠" role="题字" />
				</div>
			</article>
		</>
	);
};

const RightSideBar: Component = () => {
	return (
		<>
			<div class="w-48 self-center 2xl:self-end flex justify-end mb-6">
				<ExtendedFAB position="unset" size="small" elevation={false} text="编辑" color="primary">
					<EditIcon />
				</ExtendedFAB>
			</div>
			<TabSwitcher />
		</>
	);
};

const LeftSideBar: Component = () => {
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
				<A href="/singer/赤羽/songs">
					<Button variant="outlined" class="gap-1 items-center" size="extra-small">
						<LinkIcon class="w-5 h-5 text-xl" />
						<span>赤羽的其它歌曲</span>
					</Button>
				</A>
				<A href="/songs">
					<Button variant="outlined" class="gap-1 items-center" size="extra-small">
						<HistoryIcon class="w-5 h-5 text-xl" />
						<span>页面历史</span>
					</Button>
				</A>
			</div>
		</>
	);
};

export default function Info() {
	return (
		<Layout>
			<title>尘海绘仙缘 - 歌曲信息 - 中 V 档案馆</title>
			<div
				class="pt-8 w-full sm:w-120 sm:mx-auto lg:w-full 2xl:w-360 lg:grid lg:grid-cols-[1fr_560px_1fr]
					xl:grid-cols-[1fr_648px_1fr]"
			>
				<nav class="top-32 hidden lg:block pb-12 px-6 self-start sticky">
					<LeftSideBar />
				</nav>
				<main class="mb-24">
					<Content />
				</main>
				<div class="top-32 hidden lg:flex self-start sticky flex-col pb-12 px-6">
					<RightSideBar />
				</div>
			</div>
		</Layout>
	);
}
