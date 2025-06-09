"use client";
import { SafeMdxRenderer } from "@/lib/mdx/SafeMDX";
import "./content.css";
import remarkMdx from "remark-mdx";
import remarkGfm from "remark-gfm";
import { OptionalChidrenProps } from "@/components/ui/Dialog";
import { remark } from "remark";
import { Root } from "mdast";
import remarkCollectFootnoteDefinitions from "@/lib/mdx/footnoteHelper";
import { BackgroundDelegate } from "./Background";

const 黑幕: React.FC<OptionalChidrenProps<HTMLSpanElement>> = ({ children }) => {
	return (
		<span className="bg-on-surface dark:bg-dark-on-surface hover:text-dark-on-surface dark:hover:text-on-surface duration-200">
			{children}
		</span>
	);
};

const components = {
	黑幕: 黑幕,
	center: ({ children }: { children: React.ReactNode }) => {
		return <center>{children}</center>;
	},
	背景图片: ({ url }: { url: string }) => {
		return <BackgroundDelegate url={url} />;
	},
	poem: ({ children }: { children: React.ReactNode }) => {
		if (typeof children !== "string") {
			return <>{children}</>;
		}
		return <div className="poem" dangerouslySetInnerHTML={{ __html: children.replaceAll("\n", "<br/>") }}></div>;
	}
};

interface ContentProps {
	content: string;
}

export const ContentClient: React.FC<ContentProps> = ({ content }) => {
	try {
		const parser = remark()
			.use(remarkGfm)
			.use(remarkMdx)
			.use(remarkCollectFootnoteDefinitions)
			.use(() => {
				return (tree, file) => {
					file.data.ast = tree;
				};
			});

		const file = parser.processSync(content);
		const mdast = file.data.ast as Root;
		return (
			<div className="content">
				<SafeMdxRenderer code={content} mdast={mdast} components={components} />
			</div>
		);
	} catch (e) {
		return (
			<div className="content">
				<p className="text-on-surface-variant dark:text-dark-on-surface-variant">
					渲染出现问题。
					<br />
					错误信息: <span>{e.message}</span>
					<br />
					以下是该内容的原文：
				</p>
				<pre className="whitespace-pre-wrap">{content}</pre>
			</div>
		);
	}
};
