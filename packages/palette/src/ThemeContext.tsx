// ThemeContext.tsx
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { type ThemeMode } from "./colorTokens";

type ThemeContextType = {
	theme: ThemeMode;
	toggleTheme: () => void;
};

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const useMediaQuery = (query: string): boolean => {
	const isClient = typeof window !== "undefined";

	const [matches, setMatches] = useState<boolean>(() => {
		if (!isClient) return false;
		return window.matchMedia(query).matches;
	});

	useEffect(() => {
		if (!isClient) return;

		const mediaQueryList = window.matchMedia(query);

		if (mediaQueryList.matches !== matches) {
			setMatches(mediaQueryList.matches);
		}

		const listener = (event: MediaQueryListEvent) => setMatches(event.matches);

		mediaQueryList.addEventListener("change", listener);

		return () => mediaQueryList.removeEventListener("change", listener);
	}, [query, matches, isClient]);

	return matches;
};

export const ThemeProvider = ({ children }: { children: React.ReactNode }) => {
	const prefersDark = useMediaQuery("(prefers-color-scheme: dark)");
	const initialTheme: ThemeMode = prefersDark ? "dark" : "light";
	const [theme, setTheme] = useState<ThemeMode>(initialTheme);

	const [userToggled, setUserToggled] = useState(false);

	useEffect(() => {
		if (!userToggled) {
			setTheme(prefersDark ? "dark" : "light");
		}
	}, [prefersDark, userToggled]);

	const toggleTheme = useCallback(() => {
		setTheme((currentTheme) => (currentTheme === "light" ? "dark" : "light"));
		setUserToggled(true);
	}, []);

	const contextValue = { theme, toggleTheme };

	return <ThemeContext.Provider value={contextValue}>{children}</ThemeContext.Provider>;
};

export const useTheme = () => {
	const ctx = useContext(ThemeContext);
	if (!ctx) throw new Error("useTheme must be used inside ThemeProvider");
	return ctx;
};
