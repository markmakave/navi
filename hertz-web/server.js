const https = require('https')
const webSocket = require('ws')
const net = require('net')
const fs = require("fs")
const jpeg = require('jpeg-js')

let dataCache = new Uint8Array(0)

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

const pointWSS = new webSocket.Server({ noServer: true })
pointWSS.on('connection', (socket) => {
    console.log('[ POINT ] Client connected')

    const portionSize = 2 * 1024 * 1024
    for (let i = 0; i < dataCache.length; i += portionSize) {
        const portion = dataCache.slice(i, i + portionSize)
        socket.send(portion, { binary: true })
    }

    socket.on('close', () => {
        console.log('[ POINT ] Client disconnected')
    })
})

const videoWSS = new webSocket.Server({ noServer: true })
videoWSS.on('connection', (socket) => {
    console.log('[ VIDEO ] Client connected')    

    socket.on('close', () => {
        console.log('[ VIDEO ] Client disconnected')
    })
})

const pointUSS = net.createServer()
pointUSS.on('connection', (socket) => {
    console.log('[ POINT ] Source connected')

    socket.on('data', (data) => {
        let newData = new Uint8Array(dataCache.length + data.length)
        newData.set(dataCache)
        newData.set(data, dataCache.length)
        dataCache = newData

        pointWSS.clients.forEach((client) => {
            client.send(data, { binary: true })
        })
    })

    socket.on('close', () => {
        console.log('[ POINT ] Source disconnected')
    })
})

const videoUSS = net.createServer()
videoUSS.on('connection', (socket) => {
    console.log('[ VIDEO ] Source connected')
    
    let frameCache = new Uint8Array(0)

    socket.on('data', (data) => {
        videoWSS.clients.forEach((client) => {
            client.send(data, { binary: true })
        })
    })

    socket.on('close', () => {
        console.log('[ VIDEO ] Source disconnected')
    })
})

const port = 8080
webServer.listen(port, () => {
    console.log('[ HTTP ] Listening on port ' + port)
})

try {
    fs.unlinkSync('/tmp/hertz_points.sock')
} catch (e) {}
pointUSS.listen('/tmp/hertz_points.sock', () => {
    console.log('[ POINT ] Listening on /tmp/hertz_points.sock')
})

try {
    fs.unlinkSync('/tmp/hertz_video.sock')
}
catch (e) {}
videoUSS.listen('/tmp/hertz_video.sock', () => {
    console.log('[ VIDEO ] Listening on /tmp/hertz_video.sock')
})

webServer.on('upgrade', (req, socket, head) => {
    const pathname = req.url

    if (pathname === '/points') {
        pointWSS.handleUpgrade(req, socket, head, (ws) => {
            pointWSS.emit('connection', ws, req)
        })
    } else if (pathname === '/video') {
        videoWSS.handleUpgrade(req, socket, head, (ws) => {
            videoWSS.emit('connection', ws, req)
        })
    } else {
        socket.destroy()
    }
})
