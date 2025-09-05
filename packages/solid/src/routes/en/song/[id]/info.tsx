import { Layout } from "~/components/layout";
import { query } from "@solidjs/router";
import { Card, CardContent, CardMedia, Typography } from "@m3-components/solid";

export default function Info() {
	return (
		<Layout lang="en">
			<title></title>
			<main class="w-full pt-14 lg:max-w-lg xl:max-w-xl lg:mx-auto">
				<Card variant="outlined">
					<CardMedia
						round={false}
						src="https://i0.hdslb.com/bfs/archive/8ad220336f96e4d2ea05baada3bc04592d56b2a5.jpg"
						referrerpolicy="no-referrer"
					/>
					<CardContent>
						<div class="flex flex-col">
							<Typography.Headline variant="small">尘海绘仙缘</Typography.Headline>
							<Typography.Body class="font-medium text-on-surface-variant" variant="large">
								Chen Hai Hui Xian Yuan
							</Typography.Body>
						</div>

						<div class="mt-4 grid grid-cols-2 grid-rows-3 gap-2 ">
							<div class="flex flex-col">
								<Typography.Body class="font-semibold" variant="small">
									PUBLISHER
								</Typography.Body>
								<Typography.Body variant="large">洛凛</Typography.Body>
							</div>
							<div class="flex flex-col">
								<Typography.Body class="font-semibold" variant="small">
									DURATION
								</Typography.Body>
								<Typography.Body variant="large">4:28</Typography.Body>
							</div>
							<div class="flex flex-col">
								<Typography.Body class="font-semibold" variant="small">
									SINGER
								</Typography.Body>
								<Typography.Body variant="large">
									<a href="#">赤羽</a> <span class="text-on-surface-variant">(Chiyu)</span>
								</Typography.Body>
							</div>
							<div class="flex flex-col">
								<Typography.Body class="font-semibold" variant="small">
									PUBLISH TIME
								</Typography.Body>
								<Typography.Body variant="large">2024-12-15 12:15:00</Typography.Body>
							</div>
							<div class="flex flex-col">
								<Typography.Body class="font-semibold" variant="small">
									VIEWS
								</Typography.Body>
								<Typography.Body variant="large">12.4K (12,422)</Typography.Body>
							</div>

							<div class="flex flex-col">
								<Typography.Body class="font-semibold" variant="small">
									LINKS
								</Typography.Body>
								<Typography.Body class="flex gap-2" variant="large">
									<a href="https://www.bilibili.com/video/BV1eaq9Y3EVV/">bilibili</a>
									<a href="https://vocadb.net/S/742394">VocaDB</a>
								</Typography.Body>
							</div>
						</div>
					</CardContent>
				</Card>
			</main>
		</Layout>
	);
}
