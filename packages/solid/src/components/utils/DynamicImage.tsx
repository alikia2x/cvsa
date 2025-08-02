import { createPrefersDark } from "@solid-primitives/media";
import { Component, JSX, Match, splitProps, Switch } from "solid-js";

interface Props extends JSX.ImgHTMLAttributes<HTMLImageElement> {
	lightSrc: string;
	darkSrc: string;
}

export const DynamicImage: Component<Props> = (props) => {
	const isDark = createPrefersDark();
	const [v, rest] = splitProps(props, ["lightSrc", "darkSrc", "alt"]);
	return (
		<Switch>
			<Match when={isDark()}>
				<img src={v.darkSrc} alt={v.alt} {...rest} />
			</Match>
			<Match when={!isDark()}>
				<img src={v.lightSrc} alt={v.alt} {...rest} />
			</Match>
		</Switch>
	)
}