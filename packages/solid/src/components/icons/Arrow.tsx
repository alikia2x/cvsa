import { SVGIconComponent } from "~/components/icons/types";

export const RightArrow: SVGIconComponent = (props) => {
	return (
		<svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 28 28" {...props}>
			<path
				fill="currentColor"
				d="M18.84 15.17h-13c-.34 0-.61-.12-.84-.34-.22-.22-.33-.5-.33-.83a1.13 1.13 0 0 1 1.17-1.17h13l-3.32-3.32c-.24-.23-.35-.5-.34-.82.01-.3.12-.58.34-.81.23-.24.5-.36.83-.37.32 0 .6.1.83.34l5.34 5.33c.11.12.2.25.25.38a1.34 1.34 0 0 1 0 .88 1 1 0 0 1-.25.38l-5.34 5.33c-.23.24-.51.35-.83.34a1.19 1.19 0 0 1-1.17-1.18c0-.31.1-.58.34-.82l3.32-3.32Z"
			/>
		</svg>
	);
};
