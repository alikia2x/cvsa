module.exports = {
    apps: [
        {
            name: 'crawler-worker',
            script: 'src/worker.ts',
            cwd: './packages/api',
            interpreter: 'bun',
            instances: 1,
            autorestart: true,
            watch: false,
            max_memory_restart: '1G',
            env: {
                PATH: `${process.env.HOME}/.bun/bin:${process.env.PATH}`, // Add "~/.bun/bin/bun" to PATH
            },
        },
    ],
};
