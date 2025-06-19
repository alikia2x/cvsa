import { RouteSectionProps } from "@solidjs/router";

export default function SongLayout(props: RouteSectionProps) {
	return <div>{props.children}</div>;
}
