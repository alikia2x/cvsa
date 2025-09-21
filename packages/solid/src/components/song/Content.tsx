import { Component } from "solid-js";
import { Card, CardContent, CardMedia, Typography } from "@m3-components/solid";
import { TabSwitcher } from "~/components/song/TabSwitcher";
import { Staff } from "~/components/song/Staff";
import { SongType } from "~db/outerSchema";

export const Content: Component<{data: SongType | null}> = (props) => {
	return (
		<>
			<Card variant="outlined" class="w-full max-lg:rounded-none max-lg:border-none">
				<CardMedia
					round={false}
					src={props.data?.image || ""}
					referrerpolicy="no-referrer"
					class="relative w-full z-[2] max-lg:hidden"
				/>
				<div class="relative w-full overflow-hidden lg:hidden">
					<CardMedia
						round={false}
						src={props.data?.image || ""}
						referrerpolicy="no-referrer"
						class="relative w-full z-[2]"
					/>
					<div class="h-10 lg:h-0" />
					<CardMedia
						round={false}
						src={props.data?.image || ""}
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
						{props.data?.name}
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
				<Typography.Headline class="mx-4" variant="medium">
					简介
				</Typography.Headline>
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
				<Typography.Headline class="mx-4" variant="medium">
					制作人员
				</Typography.Headline>
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