import React, { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface DrawerProps {
	show?: boolean;
	onClose: () => void;
	children: React.ReactNode;
}

export const NavigationDrawer = ({ show = false, onClose, children }: DrawerProps) => {
	const coverRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		const handleOutsideClick = (event: MouseEvent) => {
			if (show && coverRef.current && event.target === coverRef.current) {
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
					{/* Backdrop - Fade in/out */}
					<motion.div
						ref={coverRef}
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
						className="fixed top-0 left-0 h-full bg-[#fff0ee] dark:bg-[#231918] z-50"
						style={{ width: "min(22.5rem, 70vw)" }}
						initial={{ x: -500, opacity: 0 }}
						animate={{ x: 0, opacity: 1 }}
						exit={{ x: -500, opacity: 0 }}
						transition={{ type: "spring", stiffness: 438, damping: 46 }}
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
