import { useState } from "react";
import { Layout } from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MilestoneVideos } from "@/routes/home/Milestone";
import type { Route } from "./+types/index";

export function meta({}: Route.MetaArgs) {
	return [{ title: "中V档案馆" }];
}

export default function Home() {
	const [input, setInput] = useState("");
	return (
		<Layout>
			<div className="bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 p-4 mt-8 rounded-lg">
				<h2 className="text-red-900 dark:text-red-200 text-2xl font-bold mb-2">警告</h2>
				<p>您当前看到的是档案馆用于测试的临时网站，并非正式版本。</p>
				<p>目前此网站上的内容、界面和功能可能与正式版本有很大出入，敬请注意。</p>
			</div>
			<h2 className="text-2xl font-medium mt-8 mb-4">小工具</h2>
			<div className="flex max-sm:flex-col sm:items-center justify-between gap-7 mb-8">
				<div className="flex gap-3">
					<a href="/time-calculator">
						<Button>时间计算器</Button>
					</a>
					<a href="/labelling">
						<Button>歌曲打标</Button>
					</a>
				</div>

				<div className="flex sm:w-96 gap-3">
					<Input
						placeholder="输入 BV 号或 av 号"
						value={input}
						onChange={(e) => setInput(e.target.value)}
					/>
					<a href={`/song/${input}/add`}>
						<Button>收录视频</Button>
					</a>
					<a href={`/video/${input}/info`}>
						<Button>视频信息</Button>
					</a>
				</div>
			</div>

			<MilestoneVideos />
		</Layout>
	);
}
