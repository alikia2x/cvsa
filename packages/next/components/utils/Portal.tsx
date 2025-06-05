import React from "react";
import ReactDOM from "react-dom";

export const Portal = ({ children }: { children: React.ReactNode }) => {
	const documentNotUndefined = typeof document !== "undefined";
	// Ensure portal root exists in your HTML
	const portalRoot = documentNotUndefined ? document.getElementById("portal-root") : null;

	if (!portalRoot) {
		return null;
	}

	return ReactDOM.createPortal(children, portalRoot);
};
