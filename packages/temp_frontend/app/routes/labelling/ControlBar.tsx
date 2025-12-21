import { Check, ChevronLeft, ChevronRight, X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ControlBarProps {
	currentIndex: number;
	videosLength: number;
	onPrevious: () => void;
	onNext: () => void;
	onLabel: (label: boolean) => void;
}

export function ControlBar({
	currentIndex,
	videosLength,
	onPrevious,
	onNext,
	onLabel,
}: ControlBarProps) {
	return (
		<div className="fixed bottom-0 left-0 right-0 bg-background border-t p-4 shadow-lg">
			<div className="max-w-4xl mx-auto flex items-center justify-between gap-4">
				<Button
					variant="outline"
					onClick={onPrevious}
					disabled={currentIndex === 0}
					className="flex items-center gap-2"
				>
					<ChevronLeft className="h-4 w-4" />
					上一个
				</Button>

				<div className="flex gap-4">
					<Button
						variant="destructive"
						onClick={() => onLabel(false)}
						className="flex items-center gap-2"
					>
						<X className="h-4 w-4" />否
					</Button>
					<Button
						variant="default"
						onClick={() => onLabel(true)}
						className="flex items-center gap-2 bg-green-600 hover:bg-green-700"
					>
						<Check className="h-4 w-4" />是
					</Button>
				</div>

				<Button
					variant="outline"
					onClick={onNext}
					disabled={currentIndex === videosLength - 1}
					className="flex items-center gap-2"
				>
					下一个
					<ChevronRight className="h-4 w-4" />
				</Button>
			</div>
		</div>
	);
}
