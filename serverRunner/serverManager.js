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
        let args = []
        args = [loc, JSON.stringify(param)]
        
        const pythonProcess = spawn('python', args)

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
let infiniteTraining = false
if (!token) {
    throw new Error("No BOT_TOKEN env variable set")
}
if (!trgetChatId) {
    throw new Error("No CHAT_ID env variable set")
}

const bot = new TelegramBot(token, {polling: true})
const studios = await readJsonFile("./studios.json")

console.log("Bot started")
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
    } else if(text.toLowerCase()?.startsWith("stop_all")){
        // Stops all studios
        bot.sendMessage(chatId, `Stopping all studios (0/${Object.keys(studios).length}) ...`).then(sentMsg => {lastMessageId = sentMsg.message_id;})
        let i = 1
        for(let name of Object.keys(studios)){
            const params = { action: "stop_single", credentials: JSON.stringify(studios[name])}
            const result = await runPythonScript("./serverRunner/studioManager.py", params)
            bot.editMessageText(`Studio ${studios[name].user} stopped`, {chat_id: chatId, message_id: lastMessageId})
        }
        bot.sendMessage(chatId, `All studios stopped.`)
        infiniteTraining = false
    } else if(text.toLowerCase()?.startsWith("status") && !text.toLowerCase()?.startsWith("status_all")){
        // gets status for a single studop
        const parts = text.split(" ")
        let studioName = parts[1]
        if(!studioName || Object.keys(studios).indexOf(studioName) === -1){
            bot.sendMessage(chatId, "Please provide a correct studio name. Usage: status <studio_name>")
            return
        }
        const params = { action: "status_single", credentials: JSON.stringify(studios[studioName]) }
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
        let forceNewRun = false
        if(text.toLowerCase().includes("force_new_run")){ forceNewRun = true }

        infiniteTraining = true
        while(infiniteTraining){
            bot.sendMessage(chatId, `Starting training for all studios (0/${Object.keys(studios).length}) ...`).then(sentMsg => {lastMessageId = sentMsg.message_id;})
            let i = 1
            for(let name of Object.keys(studios)){
                const params = { action: "train_single", credentials: JSON.stringify(studios[name]), forceNewRun: forceNewRun}
                runPythonScript("./serverRunner/studioManager.py", params, true) // Not awaiting it, with timed kill
                await new Promise(resolve => setTimeout(resolve, 1000))
                bot.sendMessage(chatId, `Starting studio ${studios[name].user} for training `).then(sentMsg => {lastMessageId = sentMsg.message_id;})
                i++
                await new Promise(resolve => setTimeout(resolve, 5 * 60 * 1000))
            }

            forceNewRun = false // Only for the first run
            bot.sendMessage(chatId, `All studios started for training. Waiting 4 hours before next training round...`)

            await new Promise(resolve => setTimeout(resolve, 4 * 60 * 60 * 1000)) // A 4 hour delay
        }
    } else if (text.toLowerCase()?.startsWith("train_single")){

        // gets status for a single studop
        const parts = text.split(" ")
        let studioName = parts[1]
        if(!studioName || Object.keys(studios).indexOf(studioName) === -1){
            bot.sendMessage(chatId, "Please provide a correct studio name. Usage: status <studio_name>")
            return
        }
        const params = { action: "status_single", credentials: JSON.stringify(studios[studioName]) }
        const result = await runPythonScript("./serverRunner/studioManager.py", params)

        if(result === "Stopped"){
            const params = { action: "train_single", credentials: JSON.stringify(studios[studioName]), forceNewRun: forceNewRun}
            runPythonScript("./serverRunner/studioManager.py", params, true) // Not awaiting it, with timed kill
            await new Promise(resolve => setTimeout(resolve, 1000))
            bot.sendMessage(chatId, `Starting studio ${studios[studioName].user} for training `).then(sentMsg => {lastMessageId = sentMsg.message_id;})
        } else {
            bot.sendMessage(chatId, `Studio ${studios[studioName].user} is not Stopped. Needs to be stopped to start training `)
        }
    } else if(text.toLowerCase() === "stop_training"){
        infiniteTraining = false
        bot.sendMessage(chatId, "Infinite training loop stopped. Training will not continue after active sessions are done.")
    } else if(text.toLowerCase().startsWith("training_stat")){
        const parts = text.split(" ")
        let studioName = parts[1]
        if(!studioName || Object.keys(studios).indexOf(studioName) === -1){
            bot.sendMessage(chatId, "Please provide a correct studio name. Usage: training_stat <studio_name>")
            return
        }
        const params = { action: "training_stat", credentials: JSON.stringify(studios[studioName]), botToken: token, chatId: chatId }
        await runPythonScript("./serverRunner/studioManager.py", params, true) // Not awaiting it, with timed kill
    } else if(text.toLowerCase().startsWith("upload_all_results")){
        const parts = text.split(" ")
        let studioName = parts[1]
        if(!studioName || Object.keys(studios).indexOf(studioName) === -1){
            bot.sendMessage(chatId, "Please provide a correct studio name. Usage: training_stat <studio_name>")
            return
        }
        const params = { action: "upload_results", credentials: JSON.stringify(studios[studioName]), botToken: token, chatId: chatId }
        await runPythonScript("./serverRunner/studioManager.py", params, true) // Not awaiting it, with timed kill
    } else {
        // If the command is not recognized, send a help message
        textToSend = "<b>Help: </b>\n\n" +
            "- <code>list</code>: <i>Lists all available studios </i>\n" +
            "- <code>stop single studio_name</code>: <i>Stops the specified studio </i>\n" +
            "- <code>stop_all</code>: <i>Stops all running studios </i>\n" +
            "- <code>status studio_name</code>: <i>Gets the status of the specified studio </i>\n" +
            "- <code>status_all</code>: <i>Gets the status of all studios </i>\n" +
            "- <code>train_all</code>: <i>Starts training for all studios (with 5 minutes delay between each start)  </i> \n" +
            "- <code>train_all force_new_run</code>: <i>Starts training for all studios with a forced new run in each server </i> \n" +
            "- <code>train_single studio_name optional:force_new_run</code>: <i>Starts training a specific studio </i> \n" +
            "- <code>stop_training</code>: <i>Stops further initiations of training </i> \n" +
            "- <code>training_stat studio_name</code> : <i>Gets the training status of the specified studio </i>\n" +
            "- <code>upload_all_results studio_name</code> : <i>Uploads all results from a specified studio in a separate zip file </i>\n"
        bot.sendMessage(chatId, textToSend, { parse_mode: 'HTML' })
    }
})