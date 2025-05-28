import { motion, AnimatePresence } from "framer-motion";
import React from "react";
import { TextButton } from "./Buttons/TextButton";

interface DialogProps {
	show: boolean;
	children?: React.ReactNode;
}

interface OptionalChidrenProps {
	children?: React.ReactNode;
}

type DialogHeadlineProps = OptionalChidrenProps;
type DialogSupportingTextProps = OptionalChidrenProps;
type DialogButtonGroupProps = OptionalChidrenProps;

interface DialogButtonProps extends OptionalChidrenProps {
	onClick?: React.MouseEventHandler<HTMLButtonElement>;
}

export const DialogHeadline: React.FC<DialogHeadlineProps> = ({ children }: DialogHeadlineProps) => {
	return <h2 className="text-2xl leading-8 text-on-surface dark:text-dark-on-surface">{children}</h2>;
};

export const DialogSupportingText: React.FC<DialogSupportingTextProps> = ({ children }: DialogHeadlineProps) => {
	return <div className="mt-4 text-sm leading-5 mb-6">{children}</div>;
};

export const DialogButton: React.FC<DialogButtonProps> = ({ children, onClick }: DialogButtonProps) => {
	return <TextButton onClick={onClick}>{children}</TextButton>;
};

export const DialogButtonGroup: React.FC<DialogButtonGroupProps> = ({ children }: DialogButtonGroupProps) => {
	return <div className="flex justify-end gap-2">{children}</div>;
};

export const Dialog: React.FC<DialogProps> = ({ show, children }: DialogProps) => {
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
					    shadow-2xl shadow-shadow/15 rounded-[1.75rem] p-6 dark:bg-dark-surface-container-high mx-2"
						initial={{ opacity: 0.5, transform: "scale(1.1)" }}
						animate={{ opacity: 1, transform: "scale(1)" }}
						exit={{ opacity: 0 }}
						transition={{ ease: [0.31, 0.69, 0.3, 1.02], duration: 0.3 }}
					>
						{children}
					</motion.div>
				</div>
			)}
		</AnimatePresence>
	);
};
