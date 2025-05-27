import { motion, AnimatePresence } from "framer-motion";
import React, { useEffect, useRef } from "react";
import { TextButton } from "./Buttons/TextButton";

interface DialogProps {
	show: boolean;
	onClose: () => void;
	children?: React.ReactNode;
}

interface DialogHeadlineProps {
	children?: React.ReactNode;
}

interface DialogSupportingTextProps {
	children?: React.ReactNode;
}

export const DialogHeadline: React.FC<DialogHeadlineProps> = ({ children }: DialogHeadlineProps) => {
	return <h2 className="text-2xl leading-8 text-on-surface dark:text-dark-on-surface">{children}</h2>;
};

export const DialogSupportingText: React.FC<DialogSupportingTextProps> = ({ children }: DialogHeadlineProps) => {
	return <div className="mt-4 text-sm leading-5 mb-6">{children}</div>;
};
export const Dialog: React.FC<DialogProps> = ({ show, onClose, children }: DialogProps) => {
	return (
		<AnimatePresence>
			{show && (
				<div className="w-full h-full top-0 left-0 absolute flex items-center justify-center">
					<motion.div
						className="fixed top-0 left-0 w-full h-full z-40 bg-black/20 pointer-none"
						aria-hidden="true"
						initial={{ opacity: 0 }}
						animate={{ opacity: 1 }}
						exit={{ opacity: 0 }}
						transition={{ duration: 0.3 }}
					/>
					<motion.div
						className="fixed min-w-[17.5rem] sm:max-w-[35rem] h-auto z-50 bg-surface-container-high
					    shadow-xl shadow-shadow/5 rounded-[1.75rem] p-6 dark:bg-dark-surface-container-high mx-2"
						initial={{ opacity: 0.5, transform: "scale(0.9)" }}
						animate={{ opacity: 1, transform: "scale(1)" }}
						exit={{ opacity: 0 }}
						transition={{ ease: [0.31, 0.69, 0.3, 1.02], duration: 0.3 }}
					>
						{children}
						<div className="flex justify-end gap-2">
							<TextButton onClick={onClose}>Action 1</TextButton>
							<TextButton onClick={onClose}>Action 2</TextButton>
						</div>
					</motion.div>
				</div>
			)}
		</AnimatePresence>
	);
};
