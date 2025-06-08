import { useEffect, useCallback } from "react";

export type KeyboardShortcut = {
	key: string;
	callback: () => void;
	preventDefault?: boolean;
};

export function useKeyboardShortcuts(shortcuts: KeyboardShortcut[]): void {
	const handleKeyDown = useCallback(
		(event: KeyboardEvent) => {
			shortcuts.forEach((shortcut) => {
				if (event.key === shortcut.key) {
					if (shortcut.preventDefault) {
						event.preventDefault();
					}
					shortcut.callback();
				}
			});
		},
		[shortcuts]
	);

	useEffect(() => {
		document.addEventListener("keydown", handleKeyDown);

		return () => {
			document.removeEventListener("keydown", handleKeyDown);
		};
	}, [handleKeyDown]);
}
