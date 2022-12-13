const https = require('https');
const http = require('http');
const fs = require('fs');
const net = require('net');
const WebSocket = require('ws');

const webServer = createWebServer(8080, 'ssl/privkey.pem', 'ssl/fullchain.pem');
const proxyServer = createProxyServer(8081);

const dataServer = createDataServer('/tmp/pointcloud.sock');
let dataCache = new Uint8Array(0);

const webSocketServer = createWebSocketServer(webServer, dataServer);

dataServer.on('connection', (socket) => {
    console.log('[ DATA ] Client connected');

    socket.on('data', (data) => {
        const newData = new Uint8Array(dataCache.length + data.length);
        newData.set(dataCache);
        newData.set(data, dataCache.length);
        dataCache = newData;

        webSocketServer.clients.forEach((client) => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(data, { binary: true });
            }
        });
    });

    socket.on('close', () => {
        console.log('[ DATA ] Client disconnected');
    });
});

function createDataServer(socketFilename) {
    const dataServer = net.createServer();

    try {
        fs.unlinkSync(socketFilename);
    } catch (e) {}

    dataServer.listen(socketFilename, () => {
        console.log('[ DATA ] Pointcloud server listening on ' + socketFilename);
    });

    return dataServer;
}

function createWebServer(port, key, cert) {
    const options = {
        key: fs.readFileSync(key),
        cert: fs.readFileSync(cert)
    };

    function handleApiRequest(req, res) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.write(JSON.stringify({ pointCount: dataCache.length / 16 }));
        res.end();
    }

    const webServer = https.createServer(options, (req, res) => {        

        let requestedFile = req.url;

        // get ipv4 address
        const ipv4 = req.connection.remoteAddress.split(':').pop();
        console.log('[ HTTP ] Client connectef from ' + ipv4 + ' requested ' + requestedFile); 
        
        if (requestedFile === '/') {
            requestedFile = '/index.html';
        }
    
        const fileExtension = requestedFile.split('.')[1];
    
        const contentTypeMap = {
            'html': 'text/html',
            'js': 'text/javascript',
            'css': 'text/css',
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'ico': 'image/x-icon',
            'json': 'application/json'
        };
    
        const contentType = contentTypeMap[fileExtension];
        if (!contentType) {
            console.log('[ HTTP ] Unknown file extension: ' + fileExtension);
            return;
        }

        res.writeHead(200, { 'Content-Type': contentType });
    
        requestedFile = __dirname + "/html" + requestedFile;
        try {
            const file = fs.readFileSync(requestedFile);
            res.write(file);
        } catch (e) {
            console.log('[ HTTP ] Error reading file: ' + e);
            return;
        }
    
        res.end();
    });

    webServer.on('error', (err) => {
        console.log('[ HTTP ] Web server error: ' + err);
    });

    webServer.listen(port, () => {
        console.log('[ HTTP ] Web server listening on port ' + port);
    });

    return webServer;
}

function createProxyServer(port) {
    let proxyServer = http.createServer((req, res) => {
        // get ipv4 address
        const ipv4 = req.connection.remoteAddress.split(':').pop();
        console.log('[ HTTP ] Redirecting ' + ipv4 + ' to https');
        res.writeHead(301, { 'Location': 'https://' + req.headers.host + req.url });
        res.end();
    })

    proxyServer.listen(port, () => {
        console.log('[ HTTP ] Proxy server listening on port ' + port);
    });

    return proxyServer;
}

function createWebSocketServer(webServer, dataServer) {
    const wss = new WebSocket.Server({ server: webServer });

    // send cached data to new client
    wss.on('connection', (ws) => {
        console.log('[ WEBSOCKET ] Client connected');
        ws.send(dataCache, { binary: true });
    });

    wss.on('close', () => {
        console.log('[ WEBSOCKET ] Client disconnected');
    });

    return wss;
}