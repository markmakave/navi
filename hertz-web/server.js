const https = require('https')
const webSocket = require('ws')
const net = require('net')
const fs = require("fs")
const jpegEncoder = require('jpeg-js')

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

var data = new Uint8Array(0)
const pointWSS = new webSocket.Server({ noServer: true })
pointWSS.on('connection', (socket) => {
    console.log('[ POINT ] Client connected')

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

const depthSS = net.createServer()
depthSS.on('connection', (socket) => {
    console.log('[ DEPTH ] Source connected')

    var stream = new Uint8Array(0)

    socket.on('data', (data) => {
        let newData = new Uint8Array(stream.length + data.length)
        newData.set(stream)
        newData.set(data, stream.length)
        stream = newData

        if (stream.length < 10000) {
            return
        }

        let frame = stream.slice(0, 10000)
        stream = stream.slice(10000)

        // convert grayscale image to pointcloud

        let points = new ArrayBuffer(10000 * 16)
        let width = 100
        let height = 100

        for (let i = 0; i < 10000; i++) {
            let x = 50 - i % width
            let y = 50 - Math.floor(i / width)
            let z = frame[i]

            if (z > 250){
                z = 3.40282347e+38
            }

            z /= 3

            let point = new ArrayBuffer(16)

            new Float32Array(point).set([x, y, z, 1], 0)
            new Uint8Array(point).set([255, 255, 255, 255], 12)

            new Uint8Array(points).set(new Uint8Array(point), i * 16)
        }
        
        // send points to client
        pointWSS.clients.forEach((client) => {
            client.send(points, { binary: true })
        })
        
    })

    socket.on('close', () => {
        console.log('[ DEPTH ] Source disconnected')
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

///////////////////////////////////////////////////////////////////////////

const port = 8080
webServer.listen(port, () => {
    console.log('[ HTTP ] Listening on port ' + port)
})

// listen depthSS on port 8081
depthSS.listen(8081, () => {
    console.log('[ DEPTH ] Listening on port 8081')
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
