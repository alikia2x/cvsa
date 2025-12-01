import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { HelpCircle } from "lucide-react";
import { Label } from "@/components/ui/label";

interface LabelInstructionsProps {
	open: boolean;
	onOpenChange: (open: boolean) => void;
}

export function LabelInstructions({ open, onOpenChange }: LabelInstructionsProps) {
	return (
		<div className="mt-4 mb-3">
			<div className="flex items-center">
				该视频是否包含一首<b>中V歌曲</b>？
				<Dialog open={open} onOpenChange={onOpenChange}>
					<DialogTrigger asChild>
						<Button variant="link" className="p-0 h-auto text-secondary-foreground ml-2">
							点击查看说明
						</Button>
					</DialogTrigger>
					<DialogContent className="max-w-2xl">
						<DialogHeader>
							<DialogTitle>打标说明</DialogTitle>
						</DialogHeader>
						<div className="space-y-4">
							<p>
								<b>中V歌曲</b>
								意味着它是由中文虚拟歌姬演唱，或有歌声合成的人声且歌词中包含中文。歌曲可以是原创，也可以是非原创（如翻唱、翻调等）。
							</p>
							<p>
								请根据视频信息判断是否包含中V歌曲。
								<br />
								请尽量优先参考屏幕上给出的信息，尤其是文字信息做出判断，
								<b>因为这将是模型唯一能接收的信息。</b>
								<br />
							</p>
							<p>
								<b>
									特别指示：
									<br />
								</b>
								<ul className="list-disc list-inside text-sm">
									<li>对于“AI孙燕姿”一类使用RVC等技术生成的人声，请选择“否”。</li>
									<li>对于外文歌姬（如初音未来）演唱的<b>中文歌曲</b>，请选择“是”。</li>
									<li>对于中文歌姬（如洛天依）演唱的<b>外语歌曲</b>，请选择“是”。</li>
									<li>对于自行制作的声库/歌姬（如自制UTAU/DiffSinger声库），只要歌词中有中文，请选择“是”。</li>
								</ul>
							</p>
							<p><b>如果你无法确定，请始终选择“是”。</b></p>
							<div className="bg-muted p-4 rounded-md">
								<h4 className="font-medium mb-2">键盘快捷键</h4>
								<ul className="list-disc list-inside text-sm text-secondary-foreground">
									<li>键盘左键区的按键（G、F、3、1、Q、Z 等）表示“否”</li>
									<li>键盘右键区的按键（H、J、P、8、0、L 等）表示“是”</li>
								</ul>
							</div>
						</div>
					</DialogContent>
				</Dialog>
			</div>
			<Label className="text-secondary-foreground mt-1 leading-5 block">
				中V歌曲意味着它是由中文虚拟歌姬演唱，或歌词中包含中文。歌曲可以是原创，也可以是非原创（如翻唱、翻调等）。
			</Label>
		</div>
	);
}
