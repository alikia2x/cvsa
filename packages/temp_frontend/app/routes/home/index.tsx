import { Layout } from "@/components/Layout";
import type { Route } from "./+types/index";
import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { MilestoneVideos } from "@/routes/home/Milestone";

export function meta({}: Route.MetaArgs) {
	return [{ title: "中V档案馆" }];
}

export default function Home() {
	const [input, setInput] = useState("");
	return (
		<Layout>
			<h2 className="text-2xl font-medium mt-8 mb-4">小工具</h2>
			<div className="flex max-sm:flex-col sm:items-center gap-7 mb-8">
				<a href="/time-calculator">
					<Button>时间计算器</Button>
				</a>

				<div className="flex sm:w-96 gap-3">
					<Input placeholder="输入 BV 号或 av 号" value={input} onChange={(e) => setInput(e.target.value)} />
					<a href={`/song/${input}/add`}>
						<Button>收录视频</Button>
					</a>
				</div>
			</div>

			<MilestoneVideos />
		</Layout>
	);
}
