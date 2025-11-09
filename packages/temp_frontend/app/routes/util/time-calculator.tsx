import { Layout } from "@/components/Layout";
import type { Route } from "./+types/time-calculator";
import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { formatDateTime } from "@/components/SearchResults";

export default function Home() {
	const now = new Date();
	const [time1Input, setTime1Input] = useState(formatDateTime(now));
	const [time2Input, setTime2Input] = useState(formatDateTime(now));

	const time1 = new Date(time1Input);
	const time2 = new Date(time2Input);
	const difference = time2.getTime() - time1.getTime();
	const days = Math.floor(difference / (1000 * 60 * 60 * 24));
	const hours = Math.floor(difference / (1000 * 60 * 60));
	const minutes = Math.floor((difference / (1000 * 60)) % 60);
	const diffString = `${days || 0} 天 ${hours || 0} 时 ${minutes || 0} 分`;

	return (
		<Layout>
			<h1 className="my-5 text-2xl">时间计算器</h1>
			<p>在下方输入两个时间点，即可得到两个时间点之间的时间差</p>
			<div className="flex gap-5 mt-3">
				<Input className="text-center w-50" value={time1Input} onChange={(e) => setTime1Input(e.target.value)} />
				<Input className="text-center w-50" value={time2Input} onChange={(e) => setTime2Input(e.target.value)} />
			</div>
			<p className="mt-3">{diffString}</p>
		</Layout>
	);
}
