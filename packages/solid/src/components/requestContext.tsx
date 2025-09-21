import { Accessor, Component, createSignal } from "solid-js";
import { createContext, useContext } from "solid-js";

type Hook = {
	memoizedValue: any | null;
	deps: any[] | null;
	promise: Promise<any> | null;
};

export type RequestContextValue = Map<string, Hook>;

export type Context = [Accessor<RequestContextValue>, (v: RequestContextValue) => void];

export const RequestContext = createContext<Context | null>(null);

export const RequestContextProvider: Component<{ children: any }> = (props) => {
	const initValue: RequestContextValue = new Map();
	const [value, setValue] = createSignal(initValue);
	const updateValue = (v: RequestContextValue) => {
		setValue(v);
	};

	const context: Context = [value, updateValue];

	return <RequestContext.Provider value={context}>{props.children}</RequestContext.Provider>;
};

export function useRequestContext(): Context {
	const ctx = useContext(RequestContext);
	if (!ctx) {
		throw new Error("useRequestContext must be used within a RequestContextProvider");
	}
	return ctx;
}
