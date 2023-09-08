import * as THREE from 'three';

// Scene
let scene = new THREE.Scene();

// Renderer
let renderer = new THREE.WebGLRenderer();
document.body.appendChild(renderer.domElement);
renderer.setClearColor(0x404040, 1.0);

// Camera
let camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 5);
scene.add(camera);

// Resize handler
window.onresize = function() {
	renderer.setSize(window.innerWidth, window.innerHeight);
	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
}
window.onresize();


//Handle mouse movement
window.onmousedown = function(e) {
	let lastPos = {
		x: e.clientX,
		y: e.clientY
	}
	window.onmousemove = function(e) {
		camera.position.x -= (e.clientX - lastPos.x) * 0.01;
		camera.position.y += (e.clientY - lastPos.y) * 0.01;
		lastPos = {
			x: e.clientX,
			y: e.clientY
		}
	}
}

window.onmouseup = function(e) {
	window.onmousemove = null
}

// Geometry
let geometry = new THREE.SphereGeometry(2, 100, 100);
geometry = new THREE.WireframeGeometry(geometry);
let material = new THREE.LineBasicMaterial( { color: 0xffffff, linewidth: 2 } );
let mesh = new THREE.Line( geometry, material );
scene.add(mesh);

// Lights
let light = new THREE.PointLight(0xffffff);
light.position.set(0, 0, 0);
scene.add(light);

const update = (dt) => {
	light.rotation.x += 0.1 * Math.random();
	light.rotation.y += 0.1 * Math.random();
	light.rotation.z += 0.1 * Math.random();

	let position = light.up.clone()
		.applyQuaternion(light.quaternion)
		.multiplyScalar(2);

	light.position.set(position.x, position.y, position.z);
}

// Render
let time = Date.now();
const render = () => {
	let dt = Date.now() - time;
	update(dt)
	time = Date.now();

	renderer.render(scene, camera);
	requestAnimationFrame(render);
}
render();
