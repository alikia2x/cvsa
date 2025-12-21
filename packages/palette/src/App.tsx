import "virtual:uno.css";
import type { Oklch } from "culori";
import { useAtom } from "jotai";
import { atomWithStorage } from "jotai/utils";
import { Moon, Sun } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { Buttons, Paragraph, SearchBar } from "./components/Components";
import { ColorPalette } from "./components/Palette";
import { Picker } from "./components/Picker/Picker";
import { Switch } from "./Switch";
import { useTheme } from "./ThemeContext";
import { i18nProvider } from "./utils";

const defaultColor: Oklch = { mode: "oklch", h: 29.2339, c: 0.244572, l: 0.596005 };

const colorAtom = atomWithStorage<Oklch>("selectedColor", defaultColor);
const p3Atom = atomWithStorage<boolean>("showP3", false);

function App() {
	const [useP3, setUseP3] = useAtom(p3Atom);
	const [selectedColor, setSelectedColor] = useAtom(colorAtom);
	const { theme, toggleTheme } = useTheme();

	const Icon = () => {
		if (theme === "dark") {
			return (
				<AnimatePresence>
					<motion.div
						exit={{ opacity: 0, scale: 0.6 }}
						initial={{ opacity: 0, scale: 0.6 }}
						animate={{ opacity: 1, scale: 1 }}
						onClick={toggleTheme}
						transition={{ duration: 0.5, type: "spring" }}
						className="hover:bg-black/10 dark:hover:bg-white/10 w-10 h-10 
						rounded-full flex items-center justify-center"
					>
						<Moon size={24} strokeWidth={2.5} />
					</motion.div>
				</AnimatePresence>
			);
		} else {
			return (
				<AnimatePresence>
					<motion.div
						exit={{ opacity: 0, scale: 0.6 }}
						initial={{ opacity: 0, scale: 0.6 }}
						animate={{ opacity: 1, scale: 1 }}
						onClick={toggleTheme}
						transition={{ duration: 0.5, type: "spring" }}
						className="hover:bg-black/10 dark:hover:bg-white/10 w-10 h-10 
						rounded-full flex items-center justify-center"
					>
						<Sun size={24} strokeWidth={2.5} />
					</motion.div>
				</AnimatePresence>
			);
		}
	};

	return (
		<div className="min-h-screen my-12 sm:px-6">
			<div className="max-w-7xl mx-auto">
				<h1 className="text-3xl font-bold mb-8 ml-3 text-on-background">
					CVSA Color Palette Generator
				</h1>

				<div className="grid grid-cols-1 md:grid-cols-2 lg:[grid-template-columns:2fr_3fr] xl:grid-cols-3 gap-8">
					{/* Left Column - Color Picker */}
					<div className="xl:col-span-1 sm:bg-white sm:dark:bg-zinc-800 rounded-lg shadow-sm p-3 sm:p-6">
						<h2 className="text-xl font-semibold text-on-background mb-4">
							Color Selection
						</h2>

						<div className="space-y-6">
							<div>
								<label className="block font-bold text-on-background mb-2">
									OKLCH Color Picker
								</label>
								<div className="mx-3">
									<Picker
										className="m-3"
										i18n={i18nProvider}
										useP3={useP3}
										selectedColor={selectedColor}
										onColorChange={setSelectedColor}
									/>
									<div className="flex justify-between mt-10">
										<span className="font-medium mr-2">Show P3</span>
										<Switch checked={useP3} onChange={setUseP3} />
									</div>
								</div>
							</div>

							<div>
								<label className="block font-bold text-on-background mb-2">
									Extract Colors from Image
								</label>
								<div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
									<p className="text-on-surface-variant text-sm">
										Image color extraction feature coming soon...
									</p>
								</div>
							</div>
						</div>
					</div>

					{/* Right Column */}
					<div className="xl:col-span-2 flex flex-col gap-5">
						<div className="sm:bg-white sm:dark:bg-zinc-800 rounded-lg shadow-sm p-3 sm:p-6">
							<div className="flex h-8 mb-4 justify-between items-center">
								<h2 className="text-xl font-semibold">Color Palette</h2>
								<Icon />
							</div>

							<ColorPalette baseColor={selectedColor} />
						</div>
						<div className="sm:bg-white sm:dark:bg-zinc-800 rounded-lg shadow-sm p-3 sm:p-6">
							<h2 className="text-xl font-semibold mb-6">Components</h2>
							<div className="flex flex-col gap-2">
								<SearchBar baseColor={selectedColor} />
								<Paragraph baseColor={selectedColor} />
								<Buttons baseColor={selectedColor} />
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	);
}

export default App;
