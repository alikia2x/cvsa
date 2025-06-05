import React, { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface DrawerProps {
	show?: boolean;
	onClose: () => void;
	children: React.ReactNode;
}

export const NavigationDrawer = ({ show = false, onClose, children }: DrawerProps) => {
	const scrimRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		const handleOutsideClick = (event: MouseEvent) => {
			if (show && scrimRef.current && event.target === scrimRef.current) {
				onClose();
			}
		};

		window.addEventListener("click", handleOutsideClick);
		return () => {
			window.removeEventListener("click", handleOutsideClick);
		};
	}, [show, onClose]);

	return (
		<AnimatePresence>
			{show && (
				<>
					{/* Scrim - Fade in/out */}
					<motion.div
						ref={scrimRef}
						className="fixed top-0 left-0 w-full h-full z-40 bg-black/10"
						aria-hidden="true"
						initial={{ opacity: 0 }}
						animate={{ opacity: 1 }}
						exit={{ opacity: 0 }}
						transition={{ duration: 0.3 }}
						onClick={onClose}
					/>

					{/* Drawer - Slide from left */}
					<motion.div
						className="fixed top-0 left-0 h-full bg-surface-container-low dark:bg-dark-surface-container-low
					        z-50 rounded-r-2xl"
						style={{ width: "min(22.5rem, 70vw)" }}
						initial={{ x: -500, opacity: 0 }}
						animate={{ x: 0, opacity: 1 }}
						exit={{ x: -500, opacity: 0 }}
						transition={{ duration: 0.25, ease: ["easeOut", "easeOut"] }}
						role="dialog"
						aria-modal="true"
					>
						{children}
					</motion.div>
				</>
			)}
		</AnimatePresence>
	);
};

export default NavigationDrawer;
