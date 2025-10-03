import "virtual:uno.css";
import { useEffect, useState } from "react";
import { argbFromHex, themeFromSourceColor, applyTheme } from "@material/material-color-utilities";
import { type Oklch, formatHex } from "culori";
import { i18nKeys, Picker } from "./Picker/Picker";
import { Switch } from "./Switch";

const defaultColor: Oklch = { mode: "oklch", h: 29.2339, c: 0.244572, l: 0.596005 };

const i18nProvider = (key: i18nKeys) => {
	switch (key) {
		case "l":
			return "Lightness";
		case "c":
			return "Chroma";
		case "h":
			return "Hue";
		case "fallback":
			return "Fallback";
		case "unsupported":
			return "Unavailable on this monitor";
	}
};

function App() {
	const [useP3, setUseP3] = useState(false);
	const [selectedColor, setSelectedColor] = useState<Oklch>(defaultColor);
	const colorHex = formatHex(selectedColor);

	useEffect(() => {
		const theme = themeFromSourceColor(argbFromHex(colorHex));
		const systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
		applyTheme(theme, { target: document.body, dark: systemDark });
	}, [colorHex]);

	return (
		<div className="min-h-screen my-12 mx-6">
			<div className="max-w-7xl mx-auto">
				<h1 className="text-3xl font-bold mb-8 text-on-background">CVSA Color Palette Generator</h1>

				<div className="grid grid-cols-1 md:grid-cols-2 lg:[grid-template-columns:2fr_3fr] xl:grid-cols-3 gap-8">
					{/* Left Column - Color Picker */}
					<div className="xl:col-span-1 bg-white dark:bg-zinc-800 rounded-lg shadow-sm p-6">
						<h2 className="text-xl font-semibold text-on-background mb-4">Color Selection</h2>

						<div className="space-y-6">
							<div>
								<label className="block font-bold text-on-background mb-2">OKLCH Color Picker</label>
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

					<div className="xl:col-span-2">
						<div className="bg-white dark:bg-zinc-800 rounded-lg shadow-sm p-6">
							<h2 className="text-xl font-semibold mb-6">Color Palette</h2>
						</div>
					</div>
				</div>
			</div>
		</div>
	);
}

export default App;
