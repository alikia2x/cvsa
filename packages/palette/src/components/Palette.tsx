import { Oklch } from "culori";
import { buildColorTokens } from "../colorTokens";
import { useTheme } from "../ThemeContext";
import { ColorBlock } from "./ColorBlock";

export function ColorPalette({ baseColor }: { baseColor: Oklch }) {
	const { theme } = useTheme();
	const tokens = buildColorTokens(baseColor)[theme];

	return (
		<div
			className="mx-6 grid gap-1 md:gap-4 [grid-template-columns:repeat(auto-fill,104px)] 
		md:[grid-template-columns:repeat(auto-fill,120px)] justify-between"
		>
			{Object.entries(tokens).map(([name, color]) => (
				<ColorBlock key={name} baseColor={color} text={name} />
			))}
		</div>
	);
}
