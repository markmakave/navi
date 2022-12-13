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
                gl_PointSize = 10.0;
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
}

const renderer = new PointcloudRenderer();
let consoleBody = document.getElementById("consoleBody");

let iter = 0;
function connect() {
    const socket = new WebSocket("wss://markmakave.com");

    let statusBar = document.getElementById("statusBar");

    // accept binary data on connection
    socket.binaryType = "arraybuffer";

    socket.onmessage = (event) => {
        let data = new Uint8Array(event.data);
        renderer.addPoints(data);
        let p = document.createElement("p");
        p.className = "info";
        p.innerText = "[ INFO ] Received " + data.length + " bytes of point data.";
        consoleBody.insertBefore(p, consoleBody.firstChild);
    }

    socket.onopen = () => {
        statusBar.src = "images/connected.png";
        renderer.data = new Uint8Array(0);
        renderer.pointCount = 0;

        let p = document.createElement("p");
        p.className = "info";
        p.innerText = "[ INFO ] Connected to server.";
        consoleBody.appendChild(p);
        consoleBody.insertBefore(p, consoleBody.firstChild);
    }

    socket.onclose = () => {
        statusBar.src = "images/connecting.png";
        let p = document.createElement("p");
        p.className = "error";
        p.innerText = "[ ERROR ] Connection closed. Reconnecting...";
        consoleBody.insertBefore(p, consoleBody.firstChild);
        connect();
    }
    
}

connect();

// pause on right click
let pause = false;
document.oncontextmenu = () => {
    pause = !pause;
    return false;
}

// switch renderer.mode on left click
let mode = 0;
document.onclick = () => {
    mode = (mode + 1) % 5;
    switch(mode) {
        case 0:
            renderer.mode = renderer.gl.POINTS;
            break;
        case 1:
            renderer.mode = renderer.gl.LINES;
            break;
        case 2:
            renderer.mode = renderer.gl.LINE_STRIP;
            break;
        case 3:
            renderer.mode = renderer.gl.LINE_LOOP;
            break;
        case 4:
            renderer.mode = renderer.gl.TRIANGLES;
            break;
    }
}

while (true) {
    while(pause)
    {
        renderer.render();
        await new Promise(r => setTimeout(r, 1000 / 60));
    }

    // matricies
    let model = mat4.create();
    let view = mat4.create();
    let projection = mat4.create();
    let mvp = mat4.create();

    // camera looks at origin from 5 units away in z
    mat4.lookAt(view, [0, 0, 0], [0, 0, -1], [0, 1, 0]);
    mat4.translate(view, view, [0, 0, -5]);

    // perspective projection
    mat4.perspective(projection, 45 * Math.PI / 180, renderer.canvas.width / renderer.canvas.height, 0.1, 100000);

    // model scales on each axis for breathing effect
    let scale = 1 + Math.sin(Date.now() / 1000) / 2 - 0.45;
    mat4.scale(model, model, [scale, scale, scale]);

    // model rotates on each axis using sin with a different speed and offset
    mat4.rotate(model, model, Math.sin(Date.now() / 5000 + 0) * Math.PI * 2, [1, 0, 0]);
    mat4.rotate(model, model, Math.sin(Date.now() / 5000 + 5) * Math.PI * 2, [0, 1, 0]);
    mat4.rotate(model, model, Math.sin(Date.now() / 5000 + 10) * Math.PI * 2, [0, 0, 1]);

    // multiply all matricies together
    mat4.multiply(mvp, projection, view);
    mat4.multiply(mvp, mvp, model);
    
    renderer.mvp = mvp;

    renderer.render();

    await new Promise(r => setTimeout(r, 1000 / 60));
}
