import { Root, FootnoteDefinition, Heading, List, ListItem } from "mdast";
import { Plugin } from "unified";
const remarkCollectFootnotes: Plugin<[], Root> = function () {
	return function transformer(tree) {
		const footnotes: FootnoteDefinition[] = [];

		// 收集所有 footnoteDefinition 并从树中移除它们
		tree.children = tree.children.filter((node) => {
			if (node.type === "footnoteDefinition") {
				footnotes.push(node as FootnoteDefinition);
				return false;
			}
			return true;
		});

		if (footnotes.length === 0) return tree;

		const heading: Heading = {
			type: "heading",
			depth: 2,
			children: [{ type: "text", value: "脚注" }]
		};

		const list: List = {
			type: "list",
			ordered: true,
			start: 1,
			children: footnotes.map((def, i) => {
				return {
					type: "listItem",
					children: [
						...(def.children || []).flatMap((child) => {
							return "value" in child ? { type: "text", value: child.value } : child;
						}),
						{
							type: "link",
							url: `#user-content-fnref-${i + 1}`,
							children: [
								{
									type: "text",
									value: " ↩"
								}
							]
						}
					],
					data: {
						hProperties: {
							id: `user-content-fn-${i + 1}`,
							className: "footnote-li"
						}
					}
				} as ListItem;
			})
		};

		// 添加到文档最后
		tree.children.push(heading, list);

		return tree;
	};
};

export default remarkCollectFootnotes;
