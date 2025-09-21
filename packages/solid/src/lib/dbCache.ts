import { Context } from "~/components/requestContext";

export async function useCachedFetch<T>(
	fetcher: () => Promise<T>,
	identifier: string,
	context: Context,
	deps: any[]
): Promise<T> {
	const [contextSignal, updateContext] = context;
	const hooks = contextSignal();
	let hook = hooks.get(identifier);

	if (hook && hook.promise) {
		return hook.promise;
	}

	hook = {
		memoizedValue: null,
		deps: deps,
		promise: null
	};
	const promise = fetcher().then((result) => {
		hook!.memoizedValue = result;
		hooks.set(identifier, hook!);
		updateContext(hooks);
		return result;
	});
	hook.promise = promise;
	hooks.set(identifier, hook!);
	updateContext(hooks);
	return promise;
}
