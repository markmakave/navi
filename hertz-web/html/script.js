class Camera {
    constructor(canvas) {
        this.canvas = canvas;
        this.canvas.addEventListener("mousedown", this.onMouseDown.bind(this));
        this.canvas.addEventListener("mousemove", this.onMouseMove.bind(this));
        this.canvas.addEventListener("wheel", this.onMouseWheel.bind(this));
        this.canvas.addEventListener("keydown", this.onKeyDown.bind(this));
        this.canvas.addEventListener("keyup", this.onKeyUp.bind(this));
        this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());
        this.canvas.focus();

        this.mouse = {
            locked: false
        };
        this.keys = {};

        this.position = vec3.fromValues(0, 0, 0);
        this.forward = vec3.fromValues(0, 0, 1);
        this.right = vec3.fromValues(-1, 0, 0);
        this.up = vec3.fromValues(0, 1, 0);

        this.zoom = 1;

        this.update();
    }

    update() {
        this.canvas.width = this.canvas.clientWidth;
        this.canvas.height = this.canvas.clientHeight;

        this.aspect = this.canvas.width / this.canvas.height;

        if (this.keys["w"]) {
            this.moveForward();
        }
        if (this.keys["s"]) {
            this.moveBackward();
        }
        if (this.keys["a"]) {
            this.moveLeft();
        }
        if (this.keys["d"]) {
            this.moveRight();
        }
        if (this.keys[" "]) {
            this.moveUp();
        }
        if (this.keys["Shift"]) {
            this.moveDown();
        }
    }

    onMouseDown(e) {
        if (e.button == 0) {
            if (this.mouse.locked) {
                this.mouse.locked = false;
                // unlock cursor
                document.exitPointerLock = document.exitPointerLock || document.mozExitPointerLock;
                document.exitPointerLock();

                // show cursor
                this.canvas.style.cursor = "auto";
            } else {
                this.mouse.locked = true;
                // lock cursor
                this.canvas.requestPointerLock = this.canvas.requestPointerLock || this.canvas.mozRequestPointerLock;
                this.canvas.requestPointerLock();

                // hide cursor
                this.canvas.style.cursor = "none";
            }
        }
    }

    onMouseMove(e) {
        if (this.mouse.locked) {
            let dx = -e.movementX;
            let dy = -e.movementY;

            let sensitivity = 0.002

            // rotate forward vector around up vector by dx
            let q = quat.create();
            quat.setAxisAngle(q, this.up, dx * sensitivity);
            vec3.transformQuat(this.forward, this.forward, q);

            // rotate forward vector around right vector by dy
            quat.setAxisAngle(q, this.right, dy * sensitivity);
            vec3.transformQuat(this.forward, this.forward, q);

            // update right vector
            vec3.cross(this.right, this.forward, this.up);
            vec3.normalize(this.right, this.right);

            // update forward vector
            vec3.normalize(this.forward, this.forward);
        }
    }

    onMouseWheel(e) {
        this.zoom /= Math.pow(1.001, e.deltaY);
    }

    onKeyUp(e) {
        this.keys[e.key] = false;
    }

    onKeyDown(e) {
        this.keys[e.key] = true;
    }

    moveForward() {
        vec3.add(this.position, this.position, this.forward);
    }

    moveBackward() {
        vec3.sub(this.position, this.position, this.forward);
    }

    moveLeft() {
        vec3.sub(this.position, this.position, this.right);
    }

    moveRight() {
        vec3.add(this.position, this.position, this.right);
    }

    moveUp() {
        vec3.add(this.position, this.position, this.up);
    }

    moveDown() {
        vec3.sub(this.position, this.position, this.up);
    }

    getProjectionMatrix() {
        let m = mat4.create();
        mat4.perspective(m, 45, this.aspect, 0.1, 1000);
        return m;
    }

    getViewMatrix() {
        let m = mat4.create();
        mat4.lookAt(m, this.position, vec3.add(vec3.create(), this.position, this.forward), this.up);
        mat4.scale(m, m, vec3.fromValues(this.zoom, this.zoom, this.zoom));
        return m;
    }
}


class PointcloudRenderer {
    constructor() {
        this.canvas = document.getElementById("glCanvas");

        this.gl = this.canvas.getContext("webgl");
        this.gl.clearColor(0.0, 0.0, 0.0, 0.8);
        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.depthFunc(this.gl.LEQUAL);

        // compile vertex shader
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

        // compile fragment shader
        this.fragmentShader = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(this.fragmentShader, `
            precision mediump float;
            varying vec4 vColor;
            void main() {
                gl_FragColor = vColor;
            }
        `);
        this.gl.compileShader(this.fragmentShader);

        // link shaders into program
        this.program = this.gl.createProgram();
        this.gl.attachShader(this.program, this.vertexShader);
        this.gl.attachShader(this.program, this.fragmentShader);
        this.gl.linkProgram(this.program);
        this.gl.useProgram(this.program);

////////////////////////////////////////////////////////////////////////////////////////////////////
        // payload

        this.cloud = new Uint8Array(0);

        this.cloudModel = mat4.create();
        mat4.identity(this.cloudModel);

////////////////////////////////////////////////////////////////////////////////////////////////////
        // gizmo

        this.gizmo = new ArrayBuffer(16 * 6);

        let gizmoData = new Float32Array(this.gizmo);

        // X axis
        gizmoData[0] = 0;
        gizmoData[1] = 0;
        gizmoData[2] = 0;
        new Uint8Array(this.gizmo).set([255, 0, 0, 255], 12);

        gizmoData[4] = 1;
        gizmoData[5] = 0;
        gizmoData[6] = 0;
        new Uint8Array(this.gizmo).set([255, 0, 0, 255], 28);

        // Y axis
        gizmoData[8] = 0;
        gizmoData[9] = 0;
        gizmoData[10] = 0;
        new Uint8Array(this.gizmo).set([0, 255, 0, 255], 44);

        gizmoData[12] = 0;
        gizmoData[13] = 1;
        gizmoData[14] = 0;
        new Uint8Array(this.gizmo).set([0, 255, 0, 255], 60);

        // Z axis
        gizmoData[16] = 0;
        gizmoData[17] = 0;
        gizmoData[18] = 0;
        new Uint8Array(this.gizmo).set([0, 0, 255, 255], 76);

        gizmoData[20] = 0;
        gizmoData[21] = 0;
        gizmoData[22] = 1;
        new Uint8Array(this.gizmo).set([0, 0, 255, 255], 92);

        this.gizmoModel = mat4.create();
        mat4.identity(this.gizmoModel);
        mat4.scale(this.gizmoModel, this.gizmoModel, [10, 10, 10]);

////////////////////////////////////////////////////////////////////////////////////////////////////

        // buffers
        this.gizmoBuffer = this.gl.createBuffer();
        this.pointBuffer = this.gl.createBuffer();

////////////////////////////////////////////////////////////////////////////////////////////////////

        let camera = new Camera(this.canvas);

        // set up event handlers
        window.onresize = () => {
            this.canvas.width = window.innerWidth;
            this.canvas.height = window.innerHeight;
            this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
            camera.update()
        };
        window.onresize();

        let render = () => {
            this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

            camera.update();

            // move cloudModel back and forth along the Z axis using sin wave
            mat4.identity(this.cloudModel);


            {
                // draw gizmo
                this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.gizmoBuffer);
                this.gl.bufferData(this.gl.ARRAY_BUFFER, this.gizmo, this.gl.STATIC_DRAW);

                let aPosition = this.gl.getAttribLocation(this.program, "aPosition");
                this.gl.enableVertexAttribArray(aPosition);
                this.gl.vertexAttribPointer(aPosition, 3, this.gl.FLOAT, false, 16, 0);

                let aColor = this.gl.getAttribLocation(this.program, "aColor");
                this.gl.enableVertexAttribArray(aColor);
                this.gl.vertexAttribPointer(aColor, 4, this.gl.UNSIGNED_BYTE, true, 16, 12);

                let uMVP = this.gl.getUniformLocation(this.program, "uMVP");

                let projection = camera.getProjectionMatrix();
                let view = camera.getViewMatrix();

                let mvp = mat4.create();
                mat4.multiply(mvp, projection, view);
                mat4.multiply(mvp, mvp, this.gizmoModel);

                this.gl.uniformMatrix4fv(uMVP, false, mvp);

                this.gl.drawArrays(this.gl.LINES, 0, 6);
            }

            {
                // draw cloud
                this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.pointBuffer);
                this.gl.bufferData(this.gl.ARRAY_BUFFER, this.cloud, this.gl.STATIC_DRAW);

                let aPosition = this.gl.getAttribLocation(this.program, "aPosition");
                this.gl.enableVertexAttribArray(aPosition);
                this.gl.vertexAttribPointer(aPosition, 3, this.gl.FLOAT, false, 16, 0);

                let aColor = this.gl.getAttribLocation(this.program, "aColor");
                this.gl.enableVertexAttribArray(aColor);
                this.gl.vertexAttribPointer(aColor, 4, this.gl.UNSIGNED_BYTE, true, 16, 12);

                let uMVP = this.gl.getUniformLocation(this.program, "uMVP");

                let projection = camera.getProjectionMatrix();
                let view = camera.getViewMatrix();

                let mvp = mat4.create();
                mat4.multiply(mvp, projection, view);
                mat4.multiply(mvp, mvp, this.cloudModel);

                this.gl.uniformMatrix4fv(uMVP, false, mvp);

                this.gl.drawArrays(this.gl.POINTS, 0, this.cloud.length / 16);
            }
            
            requestAnimationFrame(render);
        };

        render();
    }

    addPoints(pointArray) {
        let newData = new Uint8Array(this.cloud.length + pointArray.length);
        newData.set(this.cloud);
        newData.set(pointArray, this.cloud.length);
        this.cloud = newData;

        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.pointBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, this.cloud.buffer, this.gl.DYNAMIC_DRAW);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);
    }

    replacePoints(pointArray) {
        this.cloud = pointArray;
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.pointBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, this.cloud.buffer, this.gl.DYNAMIC_DRAW);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);
    }
}

const renderer = new PointcloudRenderer();
let consoleBody = document.getElementById("consoleBody");

let cache = new Uint8Array(0);

function connectWS() {
    const ws = new WebSocket("wss://markmakave.com/points");

    // accept binary data on connection
    ws.binaryType = "arraybuffer";

    ws.onmessage = (event) => {
        // accumulate data until we have a full frame of 24*10000 bytes
        let data = new Uint8Array(event.data);
        let newData = new Uint8Array(cache.length + data.length);
        newData.set(cache);
        newData.set(data, cache.length);
        cache = newData;

        if (cache.length < 10000) {
            return;
        }

        let frame = cache.slice(0, 10000);
        cache = cache.slice(10000);

        let width = 100
        let height = 100

        let cloud = new ArrayBuffer(width * height * 16);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let index = (y * width + x);
                let z = frame[index];

                if (z == 255) {
                    continue;
                }

                let r, g, b;

                let factor = z / 255.0;
                if (factor < 0.5) {
                    r = 255 * (1 - factor * 2);
                    g = 255 * (factor * 2);
                    b = 0;
                } else {
                    r = 0;
                    g = 255 * (1 - (factor - 0.5) * 2);
                    b = 255 * ((factor - 0.5) * 2);
                }

                new Float32Array(cloud).set([width / 2 - x, height / 2 - y, z / 3], index * 4);
                new Uint8Array(cloud).set([r, g, b, 255], index * 16 + 12);
            }
        }

        renderer.replacePoints(new Uint8Array(cloud));
    }

    ws.onopen = () => {
        renderer.data = new Uint8Array(0);

        let p = document.createElement("p");
        p.className = "info";
        p.innerText = "[ POINT ] Connected to point server.";
        consoleBody.appendChild(p);
        consoleBody.insertBefore(p, consoleBody.firstChild);
    }

    ws.onclose = () => {
        renderer.data = new Uint8Array(0);

        let p = document.createElement("p");
        p.className = "error";
        p.innerText = "[ POINT ] Connection lost. Reconnecting...";
        consoleBody.insertBefore(p, consoleBody.firstChild);
        connectWS();
    }
    
}

connectWS();
