import { formatHex, type Oklch } from "culori";
import { Copy } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import { getAdjustedColor } from "../utils";
import { Checkmark } from "./Check";

interface ColorBlockProps {
	baseColor: Oklch;
	text: string;
	l?: number;
	c?: number;
	h?: number;
}

const copy = async (text: string) => {
	await navigator.clipboard.writeText(text);
};

export const ColorBlock = ({ baseColor, text, l, c, h }: ColorBlockProps) => {
	const [hover, setHover] = useState(false);
	const [check, setCheck] = useState(false);
	const color = getAdjustedColor(baseColor, l, c, h);

	const Icon = () => {
		if (!check) {
			return (
				<AnimatePresence>
					<motion.div
						exit={{ opacity: 0 }}
						initial={{ opacity: 0 }}
						animate={{ opacity: 1 }}
					>
						<Copy size={14} strokeWidth={2.5} />
					</motion.div>
				</AnimatePresence>
			);
		} else {
			return (
				<AnimatePresence>
					<motion.div
						exit={{ opacity: 0 }}
						initial={{ opacity: 0 }}
						animate={{ opacity: 1 }}
					>
						<Checkmark width={16} height={16} strokeWidth={14} />
					</motion.div>
				</AnimatePresence>
			);
		}
	};

	return (
		<div className="w-26 md:w-30 h-36 flex flex-col items-center">
			<div
				className="w-full h-20 relative rounded-lg duration-50"
				style={{ backgroundColor: formatHex(color) }}
			/>
			<span className="mt-2 text-sm">{text}</span>
			<div
				className="flex items-center justify-center text-sm font-medium px-2 py-0.5 cursor-pointer
                    mt-1 hover:bg-gray-200 dark:hover:bg-zinc-700 rounded-md"
				onClick={() => {
					copy(formatHex(color));
					setCheck(true);
					setTimeout(() => {
						setCheck(false);
					}, 2500);
				}}
				onMouseEnter={() => setHover(true)}
				onMouseLeave={() => {
					setHover(false);
					setCheck(false);
				}}
			>
				<AnimatePresence>
					{hover && (
						<motion.div
							exit={{ opacity: 0, width: 0 }}
							initial={{ opacity: 0, width: 0 }}
							animate={{ opacity: 1, width: 22 }}
							transition={{
								opacity: { duration: 0.2, ease: "backOut" },
								width: { type: "spring", bounce: 0.2, duration: 0.5 },
							}}
						>
							<Icon />
						</motion.div>
					)}
				</AnimatePresence>
				{formatHex(color)}
			</div>
		</div>
	);
};
