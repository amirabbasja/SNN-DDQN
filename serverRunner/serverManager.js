import TelegramBot from "node-telegram-bot-api"
import {config} from 'dotenv'
import fs from 'fs/promises'
import { spawn } from 'child_process'

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
function runPythonScript(loc, param, timedKill = false) {
    return new Promise((resolve, reject) => {
        // Spawn a child process to run the Python script
        const pythonProcess = spawn('python', [loc, param.action, param.credentials])

        let studioName = JSON.parse(param.credentials).user
        let output = ''
        let errorOutput = ''
        let isResolved = false

        let timeout
        if (timedKill) {
            timeout = setTimeout(() => {
                if (!isResolved) {
                    isResolved = true
                    console.log(`Process for parameter ${param} timed out after 10 minutes, killing process...`)
                    
                    // Kill the process and all its children
                    pythonProcess.kill('SIGKILL')
                    
                    resolve(`Parameter ${param}: Process timed out after 10 minutes`)
                }
            }, 10 * 60 * 1000) // 10 minutes
        }

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
                if(timedKill){
                    clearTimeout(timeout)
                }
                isResolved = true
                
                if (code === 0) {
                    resolve(`${studioName}: ${output}`)
                } else {
                    reject(`${studioName} failed with code ${code}: ${errorOutput}`)
                }
            }
        })

        // Handle errors when starting the process
        pythonProcess.on('error', (err) => {
            if (!isResolved) {
                if(timedKill){
                    clearTimeout(timeout)
                }
                isResolved = true
                reject(`Failed to start process for ${studioName}: ${err.message}`)
            }
        })
    })
}

config({path: './serverRunner/.env'})
const token = process.env.BOT_TOKEN
const trgetChatId = process.env.CHAT_ID
if (!token) {
    throw new Error("No BOT_TOKEN env variable set")
}
if (!trgetChatId) {
    throw new Error("No CHAT_ID env variable set")
}

const bot = new TelegramBot(token, {polling: true})
const studios = await readJsonFile("./studios.json")

bot.on('message', async (msg) => {
    const chatId = msg.chat.id
    const text = msg.text
    let textToSend = ""
    let lastMessageId = null

    if(text === "list"){
        // Lists all availible studios
        const studios = await readJsonFile("./studios.json")
        textToSend = "Available Studios:\n" + Object.keys(studios).map(s => `- ${s} (${studios[s].user})`).join("\n")
        bot.sendMessage(chatId, textToSend)
    } else if(text.toLowerCase()?.startsWith("stop single")){
        // Stopps a studio
        const parts = text.split(" ")
        let studioName = parts[2]
        if(!studioName || Object.keys(studios).indexOf(studioName) === -1){
            bot.sendMessage(chatId, "Please provide a correct studio name. Usage: stop single <studio_name>")
            return
        }
        
        bot.sendMessage(chatId, `Stopping studio: ${studios[studioName].user}`).then(sentMsg => {lastMessageId = sentMsg.message_id;})
        const params = { action: "stop_single", credentials: JSON.stringify(studios[studioName])}
        const result = await runPythonScript("./serverRunner/studioManager.py", params)
        if(result.includes("not running")){
            bot.editMessageText(`Studio ${studios[studioName].user} is already stopped`, {chat_id: chatId, message_id: lastMessageId})
        } else if (!result.includes("Error") &&  result.includes("Success")){
            bot.editMessageText(`Studio ${studios[studioName].user} stopped`, {chat_id: chatId, message_id: lastMessageId})
        } else {
            bot.editMessageText(`Unknown error for stopping studio ${studios[studioName].user}: ${result}`, {chat_id: chatId, message_id: lastMessageId})
        }
    } else if(text.toLowerCase()?.startsWith("status") && !text.toLowerCase()?.startsWith("status_all")){
        // gets status for a single studop
        const parts = text.split(" ")
        let studioName = parts[1]
        if(!studioName || Object.keys(studios).indexOf(studioName) === -1){
            bot.sendMessage(chatId, "Please provide a correct studio name. Usage: status <studio_name>")
            return
        }
        const params = { action: "status_single", credentials: JSON.stringify(studios[studioName])}
        const result = await runPythonScript("./serverRunner/studioManager.py", params)
        if(result.includes("Error")){
            bot.sendMessage(chatId, `Error getting status for ${studios[studioName].user}: ${result}`)
        } else {
            bot.sendMessage(chatId, `Status for ${studios[studioName].user}: ${result}`)
        }
    } else if(text.toLowerCase()?.startsWith("status_all")){
        // Gets status for all studios
        const parts = text.split(" ")
        let noRunning = []
        let running = []
        let error = []
        bot.sendMessage(chatId, `Getting status for all studios (0/${Object.keys(studios).length}) ...`).then(sentMsg => {lastMessageId = sentMsg.message_id;})

        let i = 1
        for(let name of Object.keys(studios)){
            const params = { action: "status_single", credentials: JSON.stringify(studios[name])}
            const result = await runPythonScript("./serverRunner/studioManager.py", params)
            if(result.includes("Error")){ error.push(name) }
            else if(result.includes("Running")){ running.push(name) }
            else { noRunning.push(`${name} (${studios[name].user})`) }
            bot.editMessageText(`Getting status for all studios (${i}/${Object.keys(studios).length}) ...`, {chat_id: chatId, message_id: lastMessageId})
            i++
        }

        bot.editMessageText(`All studios status:\nNot running:\n   ${noRunning.join("\n   ")}\nRunning:\n   ${running.join("\n   ")}\nError:\n   ${error.join("\n   ")}`, {chat_id: chatId, message_id: lastMessageId})
    } else if (text.toLowerCase()?.startsWith("train_all")){
        bot.sendMessage(chatId, `Starting training for all studios (0/${Object.keys(studios).length}) ...`).then(sentMsg => {lastMessageId = sentMsg.message_id;})
        let i = 1
        for(let name of Object.keys(studios)){
            const params = { action: "train_single", credentials: JSON.stringify(studios[name])}
            runPythonScript("./serverRunner/studioManager.py", params, true) // Not awaiting it, with timed kill
            await new Promise(resolve => setTimeout(resolve, 1000))
            bot.sendMessage(chatId, `Starting studio ${studios[name].user} for training `).then(sentMsg => {lastMessageId = sentMsg.message_id;})
            i++
            await new Promise(resolve => setTimeout(resolve, 5 * 60 * 1000))
        }
    }

})