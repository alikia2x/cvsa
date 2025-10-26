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
            LOG_VERBOSE: "logs/crawler/verbose.log",
            LOG_WARN: "logs/crawler/warn.log",
            LOG_ERR: "logs/crawler/error.log"
        }
    },
    {
        name: 'crawler-filter',
        script: 'src/filterWorker.wrapper.ts',
        cwd: './packages/crawler',
        interpreter: 'bun',
        env: {
            LOG_VERBOSE: "logs/crawler/verbose.log",
            LOG_WARN: "logs/crawler/warn.log",
            LOG_ERR: "logs/crawler/error.log"
        }
    },
    {
        name: 'ml-api',
        script: 'start.py',
        cwd: './ml/api',
        interpreter: process.env.PYTHON_INTERPRETER || 'python3',
        env: {
            PYTHONPATH: './ml/api:./ml/filter',
            LOG_VERBOSE: "logs/ml/verbose.log",
            LOG_WARN: "logs/ml/warn.log",
            LOG_ERR: "logs/ml/error.log"
        }
    },
]