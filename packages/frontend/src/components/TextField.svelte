<script lang="ts">
	let focus = $state(false);
	let {
		labelText = "",
		type = "text",
		inputText = $bindable(),
		maxChar = undefined,
		supportingText = undefined,
		...rest
	} = $props();

	function onValueChange(event: Event & { currentTarget: EventTarget & HTMLInputElement }) {
		if (!event.target) return;
		const { value } = event.target as HTMLInputElement;
		inputText = value;
	}
</script>

<div {...rest}>
	<div class="relative h-14 px-4">
		<div class="absolute flex top-0 left-0 h-full w-full">
			<div class={`w-3 rounded-l-sm border-outline dark:border-dark-outline
				${(focus) ?
					"border-primary dark:border-dark-primary border-l-2 border-y-2" :
					"border-l-[1px] border-y-[1px] "}
			`}></div>

			<div class={`px-1 border-outline dark:border-dark-outline transition-none
				${(!focus && !inputText) && "border-y-[1px]"}
				${(!focus && inputText) && "border-y-[1px] border-t-0"}
				${focus && "border-primary dark:border-dark-primary border-y-2 border-t-0"}
			`}>
				<span class={`
					relative leading-6 text-base text-on-surface-variant dark:text-dark-on-surface-variant duration-150
					${(focus || inputText) ? "-top-3 text-xs leading-4" : "top-4"}
					${focus && "text-primary dark:text-dark-primary"}
				`}>
					{labelText}
				</span>
			</div>

			<div class={`flex-grow rounded-r-sm border-outline dark:border-dark-outline
					${(focus) ?
					 "border-primary dark:border-dark-primary border-r-2 border-y-2" :
					 "border-r-[1px] border-y-[1px] "}
			`}></div>
		</div>

		<input
			class="relative focus:outline-none h-full w-full"
			onfocus={() => focus = true}
			onblur={() => focus = false}
			oninput={onValueChange}
			type={type}
		/>
	</div>
	{#if supportingText || maxChar}
		<div class="w-full relative mt-1 text-on-surface-variant dark:text-dark-on-surface-variant
		 	text-xs leading-4 h-4">
			{#if supportingText}
				<span class="absolute left-4">
					{supportingText}
				</span>
			{/if}
			{#if maxChar}
				<span class="absolute right-4">
					{inputText.length}/{maxChar}
				</span>
			{/if}
		</div>
	{/if}
</div>
