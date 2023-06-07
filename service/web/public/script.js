const maxPoints = 10000

var positions = new Float32Array(maxPoints * 3);
var positionAttribute = new THREE.BufferAttribute(positions, 3);
positionAttribute.setUsage(THREE.DynamicDrawUsage);

var colors = new Uint8Array(maxPoints * 3);
var colorAttribute = new THREE.BufferAttribute(colors, 3, true);
colorAttribute.setUsage(THREE.DynamicDrawUsage);

function initRenderer() {
    // create a scene, that will hold all our elements such as objects, cameras and lights.
    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    var renderer = new THREE.WebGLRenderer();

    window.onresize = function () {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }

    // set the viewport size
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // add a camera so we can view the scene
    camera.position.x = 2;
    camera.position.y = 2;
    camera.position.z = 2;
    camera.lookAt(0, 0, 0);
    scene.add(camera);

    // add points

    var geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', positionAttribute);
    // geometry.setAttribute('color', colorAttribute);

    var material = new THREE.PointsMaterial({
        size: 2,
        sizeAttenuation: false
    })
    var points = new THREE.Points(geometry, material);
    scene.add(points);

    connectWS();

    render = function () {
        requestAnimationFrame(render);
        renderer.render(scene, camera);
    }
    render();
}

function connectWS() {
    const ws = new WebSocket('wss://markmakave.com/dashboard');

    ws.onopen = () => {
        console.log('connected');
    }

    ws.onclose = () => {
        console.log('disconnected');

        // try to reconnect
        setTimeout(() => {
            connectWS();
        })
    }

    var current = 0

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        const x = data.x;
        const y = data.y;
        const z = data.z;

        const point = new THREE.Vector3(x, y, z);
        point.toArray(positions, current * 3);

        current++
        if (current == maxPoints)
            current = 0


        positionAttribute.needsUpdate = true;

    }
}

document.addEventListener('DOMContentLoaded', function () {
    initRenderer()
    connectWS()
})
