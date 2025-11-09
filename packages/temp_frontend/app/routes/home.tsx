import { Layout } from "@/components/Layout";
import type { Route } from "./+types/home";
import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export function meta({}: Route.MetaArgs) {
	return [{ title: "中V档案馆" }];
}

export default function Home() {
	const [input, setInput] = useState("");
	return (
		<Layout>
			<h2 className="text-2xl mt-5 mb-2">小工具</h2>
			<div className="flex items-center gap-7">
				<Button>
					<a href="/util/time-calculator">时间计算器</a>
				</Button>

				<div className="flex w-96 gap-3">
					<Input placeholder="输入BV号或av号" value={input} onChange={(e) => setInput(e.target.value)} />
					<Button >
						<a href={`/song/${input}/add`}>收录视频</a>
					</Button>
				</div>
			</div>
		</Layout>
	);
}
