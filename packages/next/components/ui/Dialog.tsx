import { motion, AnimatePresence } from "framer-motion";
import React, { useRef } from "react";
import { TextButton } from "./Buttons/TextButton";
import { useEffect, useState } from "react";
import { atom, useAtom, useAtomValue, useSetAtom } from "jotai";
import { useKeyboardShortcuts } from "@/components/utils/useKeyboardEvents";
import { UAParser } from "ua-parser-js";

const focusedButtonAtom = atom(-1);

export const useDisableBodyScroll = (open: boolean) => {
	useEffect(() => {
		if (open) {
			document.body.style.overflow = "hidden";
		} else {
			document.body.style.overflow = "unset";
		}
	}, [open]);
};

export type OptionalChidrenProps<T = React.HTMLAttributes<HTMLElement>> = T & {
	children?: React.ReactNode;
};

type HeadElementAttr = React.HTMLAttributes<HTMLHeadElement>;
type DivElementAttr = React.HTMLAttributes<HTMLDivElement>;
type ButtonElementAttr = React.HTMLAttributes<HTMLButtonElement>;

type DialogHeadlineProps = OptionalChidrenProps<HeadElementAttr>;
type DialogSupportingTextProps = OptionalChidrenProps<DivElementAttr>;
type DialogButtonGroupProps = DivElementAttr & {
	children: React.ReactElement<DialogButtonProps> | React.ReactElement<DialogButtonProps>[];
	close: () => void;
};

interface DialogButtonProps extends OptionalChidrenProps<ButtonElementAttr> {
	onClick?: React.MouseEventHandler<HTMLButtonElement>;
	index?: number;
}
interface DialogProps extends OptionalChidrenProps<DivElementAttr> {
	show: boolean;
	children?: React.ReactNode;
}

export const DialogHeadline: React.FC<DialogHeadlineProps> = ({
	children,
	className,
	...rest
}: DialogHeadlineProps) => {
	return (
		<h2 className={"text-2xl leading-8 text-on-surface dark:text-dark-on-surface " + className || ""} {...rest}>
			{children}
		</h2>
	);
};

export const DialogSupportingText: React.FC<DialogSupportingTextProps> = ({
	children,
	className,
	...rest
}: DialogHeadlineProps) => {
	return (
		<div
			className={
				"mt-4 text-sm leading-5 mb-6 text-on-surface-variant dark:text-dark-on-surface-variant " + className ||
				""
			}
			{...rest}
		>
			{children}
		</div>
	);
};

export const DialogButton: React.FC<DialogButtonProps> = ({ children, onClick, index, ...rest }: DialogButtonProps) => {
	const buttonRef = useRef<HTMLButtonElement>(null);
	const focusedButton = useAtomValue(focusedButtonAtom);

	useEffect(() => {
		if (!buttonRef.current) return;
		if (focusedButton === index) buttonRef.current.focus();
	}, [focusedButton]);

	return (
		<TextButton onClick={onClick} {...rest} ref={buttonRef}>
			{children}
		</TextButton>
	);
};

export const DialogButtonGroup: React.FC<DialogButtonGroupProps> = ({
	children,
	close,
	...rest
}: DialogButtonGroupProps) => {
	const [focusedButton, setFocusedButton] = useAtom(focusedButtonAtom);
	const count = React.Children.count(children);

	useKeyboardShortcuts([
		{
			key: "Tab",
			callback: () => {
				setFocusedButton((focusedButton + 1) % count);
			},
			preventDefault: true
		},
		{
			key: "Escape",
			callback: close,
			preventDefault: true
		}
	]);

	return (
		<div className="flex justify-end gap-2" {...rest}>
			{React.Children.map(children, (child, index) => {
				if (React.isValidElement<DialogButtonProps>(child) && child.type === DialogButton) {
					return React.cloneElement(child, {
						index: index
					});
				}
				return child;
			})}
		</div>
	);
};

const useCompabilityCheck = () => {
	const [supported, setSupported] = useState(false);

	useEffect(() => {
		const parser = new UAParser(navigator.userAgent);
		const result = parser.getResult();

		const { name: browserName, version: browserVersion } = result.browser;

		let isSupported = false;

		if (!browserVersion) {
			return;
		}
		const [major] = browserVersion.split(".").map(Number);

		switch (browserName) {
			case "Chromium":
				isSupported = major >= 107;
				break;
			case "Firefox":
				isSupported = major >= 66;
				break;
			case "Safari":
				isSupported = major >= 16;
				break;
			default:
				isSupported = false;
				break;
		}

		setSupported(isSupported);
	}, []);

	return supported;
};

export const Dialog: React.FC<DialogProps> = ({ show, children, className }: DialogProps) => {
	const dialogRef = useRef<HTMLDivElement>(null);
	const contentRef = useRef<HTMLDivElement>(null);
	const setFocusedButton = useSetAtom(focusedButtonAtom);
	const isSupported = useCompabilityCheck();

	useEffect(() => {
		if (!contentRef.current || !dialogRef.current) return;

		const contentHeight = contentRef.current.offsetHeight;
		const halfSize = (contentHeight + 48) / 2;
		dialogRef.current.style.top = `calc(50% - ${halfSize}px)`;

		if (!isSupported) {
			return;
		}

		dialogRef.current.style.transition = "grid-template-rows cubic-bezier(0.05, 0.7, 0.1, 1.0) 0.35s";

		if (show) {
			dialogRef.current.style.gridTemplateRows = "1fr";
		} else {
			dialogRef.current.style.gridTemplateRows = "0.6fr";
		}
	}, [show]);

	useEffect(() => {
		setFocusedButton(-1);
	}, [show]);

	useDisableBodyScroll(show);

	return (
		<AnimatePresence>
			{show && (
				<div className="w-full h-full top-0 left-0 absolute flex justify-center">
					<motion.div
						className="fixed top-0 left-0 w-full h-full z-40 bg-black/20 pointer-none"
						aria-hidden="true"
						initial={{ opacity: 0 }}
						animate={{ opacity: 1 }}
						exit={{ opacity: 0 }}
						transition={{ duration: 0.35 }}
					/>
					<motion.div
						className={`fixed min-w-[17.5rem] sm:max-w-[35rem] h-auto z-50 bg-surface-container-high
				            shadow-2xl shadow-shadow/15 rounded-[1.75rem] p-6 dark:bg-dark-surface-container-high mx-2
							origin-top ${className} overflow-hidden grid ${isSupported && "grid-rows-[0fr]"}`}
						initial={{
							opacity: 0,
							transform: "translateY(-24px)",
							gridTemplateRows: isSupported ? undefined : "0fr"
						}}
						animate={{
							opacity: 1,
							transform: "translateY(0px)",
							gridTemplateRows: isSupported ? undefined : "1fr"
						}}
						exit={{
							opacity: 0,
							transform: "translateY(-24px)",
							gridTemplateRows: isSupported ? undefined : "0fr"
						}}
						transition={{ ease: [0.05, 0.7, 0.1, 1.0], duration: 0.35 }}
						aria-modal="true"
						ref={dialogRef}
					>
						<div className="min-h-0">
							<motion.div
								className="origin-top"
								initial={{ opacity: 0, transform: "translateY(5px)" }}
								animate={{ opacity: 1, transform: "translateY(0px)" }}
								exit={{ opacity: 0, transform: "translateY(5px)" }}
								transition={{
									ease: [0.05, 0.7, 0.1, 1.0],
									duration: 0.35
								}}
								ref={contentRef}
							>
								{children}
							</motion.div>
						</div>
					</motion.div>
				</div>
			)}
		</AnimatePresence>
	);
};
