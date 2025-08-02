import { Layout } from "~/components/shell/Layout";
import { Card, CardContent, CardMedia, Typography } from "@m3-components/solid";
import LogoLight from "/icons/zh/appbar_desktop_light.svg";
import LogoDark from "/icons/zh/appbar_desktop_dark.svg";
import { DynamicImage } from "~/components/utils/DynamicImage";

export default function Info() {
	return (
		<Layout>
			<title></title>
			<div
				class="w-full sm:w-120 sm:mx-auto lg:w-full lg:grid lg:grid-cols-[1fr_560px_minmax(300px,_1fr)]
					xl:grid-cols-[1fr_648px_1fr]"
			>
				<nav class="hidden opacity-0 pointer-events-none lg:block xl:opacity-100 xl:pointer-events-auto pt-4">
					<DynamicImage class="hidden xl:block 2xl:w-8/9" darkSrc={LogoDark} lightSrc={LogoLight} />
				</nav>
				<main class="mt-14 lg:mt-8 sm:pl-6 lg:pl-0">
					<Card variant="outlined" class="w-full">
						<CardMedia
							round={false}
							src="https://i0.hdslb.com/bfs/archive/8ad220336f96e4d2ea05baada3bc04592d56b2a5.jpg"
							referrerpolicy="no-referrer"
							class="w-full"
						/>
						<CardContent>
							<Typography.Display class="mb-3" variant="small">
								尘海绘仙缘
							</Typography.Display>
							<div class="grid grid-cols-2 grid-rows-3 gap-1">
								<Typography.Body variant="large">投稿：洛凛</Typography.Body>
								<Typography.Body variant="large">时长：4:28</Typography.Body>
								<Typography.Body variant="large">演唱：赤羽</Typography.Body>
								<Typography.Body variant="large">
									链接：
									<span class="inline-flex gap-2">
										<a href="https://www.bilibili.com/video/BV1eaq9Y3EVV/">哔哩哔哩</a>
										<a href="https://vocadb.net/S/742394">VocaDB</a>
									</span>
								</Typography.Body>
								<Typography.Body variant="large">发布时间：2024-12-15 12:15:00</Typography.Body>
								<Typography.Body variant="large">再生：1.24 万 (12,422)</Typography.Body>
							</div>
						</CardContent>
					</Card>
					<article class="mt-6">
						<Typography.Headline variant="medium">简介</Typography.Headline>
						<Typography.Body class="mt-2" variant="large">
							<span class="font-medium">《尘海绘仙缘》</span>是<a href="#">洛凛</a>于
							<span>
								&VeryThinSpace;2024&VeryThinSpace;年&VeryThinSpace;12&VeryThinSpace;月&VeryThinSpace;15&VeryThinSpace;日
							</span>
							投稿至
							<a href="#">哔哩哔哩</a>的&ThinSpace;<a href="#">Synthesizer V</a>&ThinSpace;
							<span>中文</span>
							<span>原创歌曲</span>, 由<a href="#">赤羽</a>演唱。
						</Typography.Body>
					</article>
				</main>
				<nav></nav>
			</div>
		</Layout>
	);
}
