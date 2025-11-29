import { Layout } from "@/components/Layout";
import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { formatDateTime } from "@/components/SearchResults";

export default function Home() {
	const now = new Date();
	const [time1Input, setTime1Input] = useState(formatDateTime(now));
	const [time2Input, setTime2Input] = useState(formatDateTime(now));

	const time1 = new Date(time1Input);
	const time2 = new Date(time2Input);
	const difference = time2.getTime() - time1.getTime();
	const days = Math.floor(difference / (1000 * 60 * 60 * 24));
	const hours = Math.floor(difference / (1000 * 60 * 60)) % 24;
	const minutes = Math.floor((difference / (1000 * 60)) % 60);
	const seconds = Math.floor((difference / 1000) % 60);

	const diffString = `${Math.abs(days) || 0} 天 ${Math.abs(hours) || 0} 时 ${Math.abs(minutes) || 0} 分`;
	const isNegative = difference < 0;

	const setQuickTime = (hoursOffset: number) => {
		const newTime = new Date(time1Input);
		newTime.setHours(newTime.getHours() + hoursOffset);
		setTime2Input(formatDateTime(newTime));
	};

	return (
		<Layout>
			<div className="max-w-4xl mx-auto space-y-6">
				<div className="space-y-2 mt-8">
					<h2 className="text-2xl font-bold tracking-tight">时间计算器</h2>
					<p className="text-muted-foreground">输入两个时间点，计算精确的时间差</p>
				</div>

				{/* 时间输入区域 */}
				<Card>
					<CardHeader>
						<CardTitle>时间设置</CardTitle>
						<CardDescription>选择或输入开始时间和结束时间</CardDescription>
					</CardHeader>
					<CardContent className="space-y-6">
						<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
							<div className="space-y-3">
								<Label htmlFor="start-time">开始时间</Label>
								<Input
									id="start-time"
									value={time1Input}
									onChange={(e) => setTime1Input(e.target.value)}
								/>
								<Button
									variant="outline"
									size="sm"
									onClick={() => setTime1Input(formatDateTime(now))}
									className="w-full"
								>
									设为当前时间
								</Button>
							</div>

							<div className="space-y-3">
								<Label htmlFor="end-time">结束时间</Label>
								<Input
									id="end-time"
									value={time2Input}
									onChange={(e) => setTime2Input(e.target.value)}
								/>
								<Button
									variant="outline"
									size="sm"
									onClick={() => setTime2Input(formatDateTime(now))}
									className="w-full"
								>
									设为当前时间
								</Button>
							</div>
						</div>
					</CardContent>
				</Card>

				<Card>
					<CardHeader>
						<CardTitle>计算结果</CardTitle>
						<CardDescription>两个时间点之间的精确时间差</CardDescription>
					</CardHeader>
					<CardContent>
						<div className="text-center space-y-4">
							<div className="space-y-2">
								<div className="text-2xl font-bold">{isNegative ? "时间差为负值" : diffString}</div>
								{isNegative && (
									<div className="text-lg text-muted-foreground">结束时间早于开始时间</div>
								)}
							</div>

							<div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4">
								<div className="space-y-2">
									<div className="text-2xl font-bold">{Math.abs(days)}</div>
									<div className="text-sm text-muted-foreground">天数</div>
								</div>
								<div className="space-y-2">
									<div className="text-2xl font-bold">{Math.abs(hours)}</div>
									<div className="text-sm text-muted-foreground">小时</div>
								</div>
								<div className="space-y-2">
									<div className="text-2xl font-bold">{Math.abs(minutes)}</div>
									<div className="text-sm text-muted-foreground">分钟</div>
								</div>
								<div className="space-y-2">
									<div className="text-2xl font-bold">{Math.abs(seconds)}</div>
									<div className="text-sm text-muted-foreground">秒数</div>
								</div>
							</div>
						</div>
					</CardContent>
				</Card>
			</div>
		</Layout>
	);
}
