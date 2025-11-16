import 'dotenv/config'

export const apps = [
    {
        name: 'crawler-jobadder',
        script: 'src/jobAdder.wrapper.ts',
        cwd: './packages/crawler',
        interpreter: 'bun',
    },
    {
        name: 'crawler-worker',
        script: 'src/worker.ts',
        cwd: './packages/crawler',
        interpreter: 'bun',
        env: {
            LOG_VERBOSE: "logs/verbose.log",
            LOG_WARN: "logs/warn.log",
            LOG_ERR: "logs/error.log"
        }
    },
    {
        name: 'crawler-filter',
        script: 'src/filterWorker.wrapper.ts',
        cwd: './packages/crawler',
        interpreter: 'bun',
        env: {
            LOG_VERBOSE: "logs/verbose.log",
            LOG_WARN: "logs/warn.log",
            LOG_ERR: "logs/error.log"
        }
    },
    {
        name: 'ml-api',
        script: 'start.py',
        cwd: './ml/api',
        interpreter: process.env.PYTHON_INTERPRETER || 'python3',
        env: {
            PYTHONPATH: './ml/api:./ml/filter',
            LOG_VERBOSE: "logs/verbose.log",
            LOG_WARN: "logs/warn.log",
            LOG_ERR: "logs/error.log"
        }
    },
    {
        name: 'cvsa-be',
        script: 'src/index.ts',
        cwd: './packages/backend',
        interpreter: 'bun',
        env: {
            NODE_ENV: 'production'
        }
    },
]