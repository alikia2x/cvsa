import type { SVGProps } from "react";

export const Checkmark: React.FC<SVGProps<SVGSVGElement>> = (props) => {
	return (
		<svg viewBox="10 5 90 85" xmlns="http://www.w3.org/2000/svg" {...props}>
			<path
				fill="none"
				stroke="currentColor"
				strokeWidth={props.strokeWidth || 14}
				strokeLinecap="round"
				strokeLinejoin="round"
				d="M25 50 L45 70 L75 30"
				strokeDasharray="100"
				strokeDashoffset="100"
				style={{
					animation: "draw 0.3s forwards ease-out",
				}}
			/>
			<style>
				{`
				@keyframes draw {
					to {
						stroke-dashoffset: 0;
					}
				}
				`}
			</style>
		</svg>
	);
};
