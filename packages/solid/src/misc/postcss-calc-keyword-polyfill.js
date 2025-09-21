import { plugin } from "postcss";

export default plugin("postcss-calc-keyword-polyfill", () => {
	const replacements = {
		pi: "3.141592653589793",
		e: "2.718281828459045",
		infinity: "1e308", // A very large number to simulate infinity
		"-infinity": "-1e308", // A very small number to simulate -infinity
		nan: "0/0" // Division by zero in calc() results in NaN in modern browsers
	};

	// Regex to find the keywords, case-insensitive
	const keywordRegex = new RegExp(`\\b(-?(${Object.keys(replacements).join("|")}))\\b`, "gi");

	return (root) => {
		root.walkDecls((decl) => {
			// Check if the declaration value contains calc()
			if (decl.value.toLowerCase().includes("calc(")) {
				decl.value = decl.value.replace(/calc\(([^)]+)\)/gi, (match, expression) => {
					const newExpression = expression.replace(keywordRegex, (keyword) => {
						const lowerKeyword = keyword.toLowerCase();
						if (lowerKeyword in replacements) {
							return replacements[lowerKeyword];
						}
						// Handle cases like -pi and -e
						if (lowerKeyword.startsWith("-") && lowerKeyword.substring(1) in replacements) {
							return `-${replacements[lowerKeyword.substring(1)]}`;
						}
						return keyword; // Should not happen with the current regex, but as a fallback
					});
					return `calc(${newExpression})`;
				});
			}
		});
	};
});
