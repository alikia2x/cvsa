import { SVGIconComponent } from "./types";

export const LinkIcon: SVGIconComponent = (props) => {
	return (
		<svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24" {...props}>
			<path
				fill="currentColor"
				d="M11 17H7a4.82 4.82 0 0 1-3.54-1.46A4.82 4.82 0 0 1 2 12c0-1.38.49-2.56 1.46-3.54A4.82 4.82 0 0 1 7 7h4v2H7c-.83 0-1.54.3-2.13.88A2.9 2.9 0 0 0 4 12c0 .83.3 1.54.88 2.13.58.58 1.29.87 2.12.87h4v2Zm-3-4v-2h8v2H8Zm5 4v-2h4c.83 0 1.54-.3 2.13-.88.58-.58.87-1.29.87-2.12 0-.83-.3-1.54-.88-2.13A2.9 2.9 0 0 0 17 9h-4V7h4c1.38 0 2.56.49 3.54 1.46A4.82 4.82 0 0 1 22 12c0 1.38-.49 2.56-1.46 3.54A4.8 4.8 0 0 1 17 17h-4Z"
			/>
		</svg>
	);
};
