import fs from 'fs/promises'
import { spawn } from 'child_process';

async function readJsonFile(path) {
    try {
        const data = await fs.readFile(path, 'utf8')
        const jsonData = JSON.parse(data) // Parse JSON string to object
        return jsonData
    } catch (err) {
        throw new Error('Error:' + err)
    }
}


// Function to run a Python script with a given parameter and 10-minute timeout
function runPythonScript(param) {
    return new Promise((resolve, reject) => {
        // Spawn a child process to run the Python script
        const pythonProcess = spawn('python', ['./serverRunner/stopStudio.py', param])

        let output = ''
        let errorOutput = ''
        let isResolved = false

        // Set up 10-minute timeout (600,000 milliseconds)
        const timeout = setTimeout(() => {
            if (!isResolved) {
                isResolved = true
                console.log(`Process for parameter ${param} timed out after 10 minutes, killing process...`)
                
                // Kill the process and all its children
                pythonProcess.kill('SIGKILL')
                
                resolve(`Parameter ${param}: Process timed out after 10 minutes`)
            }
        }, 10 * 60 * 1000) // 10 minutes

        // Capture standard output
        pythonProcess.stdout.on('data', (data) => {
            output += data.toString()
        })

        // Capture standard error
        pythonProcess.stderr.on('data', (data) => {
            errorOutput += data.toString()
        })

        // Handle process exit
        pythonProcess.on('close', (code) => {
            if (!isResolved) {
                isResolved = true
                clearTimeout(timeout) // Clear the timeout since process finished normally
                
                if (code === 0) {
                    resolve(`Parameter ${param}: ${output}`)
                } else {
                    reject(`Parameter ${param} failed with code ${code}: ${errorOutput}`)
                }
            }
        })

        // Handle errors when starting the process
        pythonProcess.on('error', (err) => {
            if (!isResolved) {
                isResolved = true
                clearTimeout(timeout)
                reject(`Failed to start process for parameter ${param}: ${err.message}`)
            }
        })
    })
}

const studios = await readJsonFile("./studios.json")

for(let studioName in studios){
    console.log("Stopping studio", studios[studioName].user)
    runPythonScript(JSON.stringify(studios[studioName]))
    await new Promise(resolve => setTimeout(resolve, 10000))
}