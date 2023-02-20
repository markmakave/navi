const https = require('https')
const webSocket = require('ws')
const net = require('net')
const fs = require("fs")

const options = {
    key: fs.readFileSync('ssl/privkey.pem'),
    cert: fs.readFileSync('ssl/fullchain.pem')
}

const webServer = https.createServer(options, (req, res) => {
    // file format map
    const fileFormatMap = {
        'js': 'text/javascript',
        'html': 'text/html',
        'css': 'text/css',
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'ico': 'image/x-icon',
        'json': 'application/json',
        'gif': 'image/gif',
    }
    
    // get ipv4 address
    const ipv4 = req.connection.remoteAddress.split(':').pop()
    console.log('[ HTTP ] Client connected from ' + ipv4 + ' requested ' + req.url)

    let requestedFile = req.url
    if (requestedFile === '/') {
        requestedFile = '/index.html'
    }

    // get file extension
    const fileExtension = requestedFile.split('.')[1]

    // get content type
    if (!fileFormatMap[fileExtension]) {
        console.log('[ HTTP ] Unknown file extension: ' + fileExtension)
        return
    }

    const contentType = fileFormatMap[fileExtension]
    res.writeHead(200, { 'Content-Type': contentType })

    requestedFile = __dirname + '/html' + requestedFile
    try {
        const file = fs.readFileSync(requestedFile)
        res.write(file)
    } catch (e) {
        console.log('[ HTTP ] Error reading file: ' + e)
    }

    res.end()
})

const wss = new webSocket.Server({ noServer: true })
wss.on('connection', (socket) => {
    console.log('[ WEBSOCKET ] Client connected')

    socket.on('close', () => {
        console.log('[ WEBSOCKET ] Client disconnected')
    })
})

const ss = net.createServer()
ss.on('connection', (socket) => {
    console.log('[ SOCKET ] Source connected')

    socket.on('data', (data) => {
        // forward all data to websocket clients
	wss.clients.forEach((client) => {
            client.send(data, { binary: true })
        })
    })

    socket.on('close', () => {
        console.log('[ SOCKET ] Source disconnected')
    })
})

///////////////////////////////////////////////////////////////////////////

const port = 8080
webServer.listen(port, () => {
    console.log('[ HTTP ] Listening on port ' + port)
})

// listen depthSS on port 8081
ss.listen(8081, () => {
    console.log('[ SOCKET ] Listening on port 8081')
})

webServer.on('upgrade', (req, socket, head) => {
    const pathname = req.url

    if (pathname === '/points') {
        wss.handleUpgrade(req, socket, head, (ws) => {
            wss.emit('connection', ws, req)
        })
    } else {
        socket.destroy()
    }
})
