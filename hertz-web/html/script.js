class PointcloudRenderer {
    constructor() {
        this.canvas = document.getElementById("glCanvas");

        this.gl = this.canvas.getContext("webgl");
        this.gl.clearColor(0.0, 0.0, 0.0, 0.8);
        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.depthFunc(this.gl.LEQUAL);
        this.mode = this.gl.POINTS;

        this.vertexShader = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(this.vertexShader,
            `
            attribute vec3 aPosition;
            attribute vec4 aColor;
            varying vec4 vColor;
            uniform mat4 uMVP;
            void main() {
                gl_Position = uMVP * vec4(aPosition, 1.0);
                gl_PointSize = 5.0;
                vColor = aColor;
            }
            `
        );
        this.gl.compileShader(this.vertexShader);

        this.fragmentShader = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(this.fragmentShader, `
            precision mediump float;
            varying vec4 vColor;
            void main() {
                gl_FragColor = vColor;
            }
        `);
        this.gl.compileShader(this.fragmentShader);

        this.program = this.gl.createProgram();
        this.gl.attachShader(this.program, this.vertexShader);
        this.gl.attachShader(this.program, this.fragmentShader);
        this.gl.linkProgram(this.program);
        this.gl.useProgram(this.program);

        this.data = new Uint8Array(0);
        this.pointCount = 0;

        this.pointBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.pointBuffer);
        let aPositionLocation = this.gl.getAttribLocation(this.program, "aPosition");
        this.gl.enableVertexAttribArray(aPositionLocation);
        this.gl.vertexAttribPointer(aPositionLocation, 3, this.gl.FLOAT, false, 16, 0);
        let aColorLocation = this.gl.getAttribLocation(this.program, "aColor");
        this.gl.enableVertexAttribArray(aColorLocation);
        this.gl.vertexAttribPointer(aColorLocation, 4, this.gl.UNSIGNED_BYTE, true, 16, 12);

        this.mvp = mat4.create();
        this.mvpLocation = this.gl.getUniformLocation(this.program, "uMVP");

        window.onresize = () => {
            this.canvas.width = window.innerWidth;
            this.canvas.height = window.innerHeight;
            this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
            mat4.lookAt(this.mvp, [0, 0, 0], [0, 0, -1], [0, 1, 0]);
            mat4.perspective(this.mvp, 45 * Math.PI / 180, this.canvas.width / this.canvas.height, 0.1, 100000);
            mat4.translate(this.mvp, this.mvp, [0, 0, -5]);
        };
        window.onresize();
    }

    render() {
        this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

        this.gl.uniformMatrix4fv(this.mvpLocation, false, this.mvp);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.pointBuffer);

        this.gl.drawArrays(this.mode, 0, this.pointCount);

    }

    addPoints(pointArray) {
        let newData = new Uint8Array(this.data.length + pointArray.length);
        newData.set(this.data);
        newData.set(pointArray, this.data.length);
        this.data = newData;

        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.pointBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, this.data.buffer, this.gl.DYNAMIC_DRAW);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);
        
        this.pointCount = this.data.length / 16;
    }

    replacePoints(pointArray) {
        this.data = pointArray;
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.pointBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, this.data.buffer, this.gl.DYNAMIC_DRAW);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);

        this.pointCount = this.data.length / 16;
    }
}

const renderer = new PointcloudRenderer();
let consoleBody = document.getElementById("consoleBody");
let stereoFrame = document.getElementById("stereoFrame");

stereoFrame.src = "images/na.png";

let cache = new Uint8Array(0);

function connectPWS() {
    const pointWS = new WebSocket("wss://markmakave.com/points");

    // accept binary data on connection
    pointWS.binaryType = "arraybuffer";

    pointWS.onmessage = (event) => {
        // accumulate data until we have a full frame of 24*10000 bytes
        let data = new Uint8Array(event.data);
        let newData = new Uint8Array(cache.length + data.length);
        newData.set(cache);
        newData.set(data, cache.length);
        cache = newData;

        if (cache.length < 16*10000) {
            return;
        }

        let points = cache.slice(0, 16*10000);
        cache = cache.slice(16*10000);

        renderer.replacePoints(points);
    }

    pointWS.onopen = () => {
        renderer.data = new Uint8Array(0);
        renderer.pointCount = 0;

        let p = document.createElement("p");
        p.className = "info";
        p.innerText = "[ POINT ] Connected to point server.";
        consoleBody.appendChild(p);
        consoleBody.insertBefore(p, consoleBody.firstChild);
    }

    pointWS.onclose = () => {
        let p = document.createElement("p");
        p.className = "error";
        p.innerText = "[ POINT ] Connection lost. Reconnecting...";
        consoleBody.insertBefore(p, consoleBody.firstChild);
        connectPWS();
    }
    
}

function connectVWS() {
    const videoWS = new WebSocket("wss://markmakave.com/video");

    videoWS.willReadFrequently = true;
    videoWS.binaryType = "arraybuffer";

    videoWS.onmessage = (event) => {
        let data = new Uint8Array(event.data);
        let blob = new Blob([data], {type: "image/jpeg"});
        stereoFrame.src = URL.createObjectURL(blob);
    }

    videoWS.onopen = () => {
        let p = document.createElement("p");
        p.className = "info";
        p.innerText = "[ VIDEO ] Connected to video server.";
        consoleBody.insertBefore(p, consoleBody.firstChild);
    }

    videoWS.onclose = () => {
        let p = document.createElement("p");
        p.className = "error";
        p.innerText = "[ VIDEO ] Connection lost. Reconnecting...";
        consoleBody.insertBefore(p, consoleBody.firstChild);
        stereoFrame.src = "images/na.png";
        connectVWS();
    }
}

connectPWS();
connectVWS();

// switch renderer.mode on left click
let mode = 0;
document.onclick = () => {
    mode = (mode + 1) % 3;
    switch(mode) {
        case 0:
            renderer.mode = renderer.gl.POINTS;
            break;
        case 1:
            renderer.mode = renderer.gl.LINE_STRIP;
            break;
        case 2:
            renderer.mode = renderer.gl.TRIANGLES;
            break;
    }
}

while (true) {
    let model = mat4.create();
    let view = mat4.create();
    let projection = mat4.create();
    let mvp = mat4.create();

    mat4.lookAt(view, [0, 0, -50], [0, 0, 0], [0, 1, 0]);
    mat4.perspective(projection, 90 * Math.PI / 180, renderer.canvas.width / renderer.canvas.height, 0.1, 100000);

    mat4.multiply(mvp, projection, view);
    mat4.multiply(mvp, mvp, model);


    
    renderer.mvp = mvp;

    renderer.render();

    await new Promise(r => setTimeout(r, 1000 / 60));
}
