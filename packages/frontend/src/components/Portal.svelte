<script>
	let ref;
	$: ref && document.body.appendChild(ref);
</script>

<div bind:this={ref}>
	<slot></slot>
</div>