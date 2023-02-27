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

        this.cloud = new Uint8Array(0)
        this.indices = new Uint16Array(0)

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

        this.gizmoModel = mat4.create()
        mat4.identity(this.gizmoModel)
        mat4.scale(this.gizmoModel, this.gizmoModel, [10, 10, 10])

////////////////////////////////////////////////////////////////////////////////////////////////////

        // buffers
        this.gizmoBuffer = this.gl.createBuffer()
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.gizmoBuffer)
        this.gl.bufferData(this.gl.ARRAY_BUFFER, this.gizmo, this.gl.STATIC_DRAW)
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null)

        this.pointBuffer = this.gl.createBuffer();
        this.indexBuffer = this.gl.createBuffer()

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
                this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer)

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

                this.gl.drawElements(this.gl.TRIANGLES, this.indices.length, this.gl.UNSIGNED_SHORT, 0);
            }
            
            requestAnimationFrame(render);
        };

        render();
    }

    setPoints(pointArray) {
        this.cloud = pointArray;
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.pointBuffer)
        this.gl.bufferData(this.gl.ARRAY_BUFFER, this.cloud, this.gl.STATIC_DRAW);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null)
    }

    setIndices(indexArray) {
        this.indices = indexArray;
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer)
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, this.indices.buffer, this.gl.STATIC_DRAW)
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, null)
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

        let pair = convert(frame);

        let vertex_array = pair[0]
        let index_array = pair[1]

        renderer.setPoints(vertex_array);
        renderer.setIndices(index_array);
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

////////////////////////////////////////////////////////////////////////////////////////////////////

const width = 100
const height = 100

const value_offset = 35

function convert(buffer) {

    let vertex_array = new ArrayBuffer(width * height * 16)
    let index_array = new Uint16Array(99 * 99 * 2 * 3)
    let index_count = 0

    for (var y = 0; y < height; ++y) {
        for (var x = 0; x < width; ++x) {

            let value = buffer[y * width + x]

            // process vertex

            let cord = {
                x: x - width / 2.0,
                y: y - height / 2.0,
                z: value / 10
            }

            let color = {
                r: 0,
                g: 0,
                b: 0,
                a: 255
            }

            // set color to gradiet based on factor:
            // 1.0 - blue
            // 0.5 - green
            // 0.0 - red
            let factor = value / (255.0 - value_offset)

            if (factor > 0.5) {
                color.b = 255 * (factor * 2 - 1)
                color.g = 255 * (2 - 2 * factor)
            } else {
                color.g = 255 * (factor * 2)
                color.r = 255 * (1 - factor * 2)
            }

            // append cords and color to vectex_array
            new Float32Array(vertex_array).set([cord.x, cord.y, cord.z], (y * width + x) * 4)
            new Uint8Array(vertex_array).set([color.r, color.g, color.b, color.a], (y * width + x) * 16 + 12)

            // process index array
            // create a triangle for every 4 close vertex
            if (x > 0 && y > 0) {
                let base = new Float32Array(vertex_array)
        
                let neigbour = [
                    base[((y-1) * width + (x-1)) * 4 + 2],
                    base[((y-1) * width + (x-0)) * 4 + 2],
                    base[((y-0) * width + (x-1)) * 4 + 2],
                    base[((y-0) * width + (x-0)) * 4 + 2]
                ]

                if (neigbour[0] != 255) {
                    if (neigbour[3] != 255) {
                        if (neigbour[1] != 255) {
                            // append 
                            // [
                            //     (y-1) * width + (x-1), 
                            //     (y-1) * width + (x-0),
                            //     (y-0) * width + (x-0)
                            // ]
                            index_array.set([
                                (y-1) * width + (x-1),
                                (y-1) * width + (x-0),
                                (y-0) * width + (x-0),
                            ], index_count)
                            index_count += 3
                        }

                        if (neigbour[2] != 255) {
                            // append 
                            // [
                            //     (y-1) * width + (x-1), 
                            //     (y-0) * width + (x-1),
                            //     (y-0) * width + (x-0)
                            // ]
                            index_array.set([
                                (y-1) * width + (x-1),
                                (y-0) * width + (x-1),
                                (y-0) * width + (x-0),
                            ], index_count)
                            index_count += 3
                        }
                    } else {
                        if (neigbour[1] != 255 && neigbour[2] != 255) {
                            // append 
                            // [
                            //     (y-1) * width + (x-1), 
                            //     (y-1) * width + (x-0),
                            //     (y-0) * width + (x-1)
                            // ]
                            index_array.set([
                                (y-1) * width + (x-1),
                                (y-1) * width + (x-0),
                                (y-0) * width + (x-1),
                            ], index_count)
                            index_count += 3
                        } else {
                            // no way to build a triangle
                            continue;
                        }
                    }
                } else {
                    if (neigbour[3] != 255) {
                        if (neigbour[1] != 255 && neigbour[2] != 255) {
                            // append 
                            // [
                            //     (y-1) * width + (x-0), 
                            //     (y-0) * width + (x-0),
                            //     (y-0) * width + (x-1)
                            // ]
                            index_array.set([
                                (y-1) * width + (x-0),
                                (y-0) * width + (x-0),
                                (y-0) * width + (x-1),
                            ], index_count)
                            index_count += 3
                        } else {
                            // no way to build a triangle
                            continue;
                        }
                    } else {
                        // no way to build a triangle
                        continue;
                    }
                }
            }
        }
    }

    index_array = index_array.slice(0, index_count)

    return [vertex_array, index_array]
}

////////////////////////////////////////////////////////////////////////////////////////////////////
