var renderer, camera;
var scene, element;
var ambient;
var aspectRatio, windowHalf;
var mouse, time;

var controls;
var clock;
var grid = [];
var baked = [];

var DataValues = {
	mapData: [],
    xSize: 0,
    ySize: 0,
    zSize: 0,
    needsUpdate: false,
    iter: 0,
    maxIters: 0,
    autoPlay: false,
    playSpeed: 4.5,         //Seconds between iterations
    invert: true,
    autoRotate: true,
    rotationSpeed: 20,
    meshMaterial: undefined,
    next: function() { 
        DataValues.iter = (DataValues.iter + 1) % DataValues.maxIters; 
        DataValues.needsUpdate = true;
        },
    prev: function() { 
        DataValues.iter = (--DataValues.iter < 0) ? DataValues.maxIters - 1 : DataValues.iter; 
        DataValues.needsUpdate = true;
        }
};

function init() {
    window.addEventListener('resize', onResize, false);

    initScene();
    initGrid();

	stats = new Stats();
	stats.domElement.style.position = 'absolute';
	stats.domElement.style.top = '0px';
	stats.domElement.style.zIndex = 100;

    element = document.getElementById('viewport');
	element.appendChild( stats.domElement );

    var gui = new dat.GUI();
    var iterController = gui.add(DataValues, 'iter', 0, DataValues.maxIters-1);
    iterController.listen();
    iterController.onChange( function (value) {
        if(DataValues.iter >= DataValues.maxIters) {
            DataValues.iter = DataValues.maxIters - 1;
        } 
        else if(DataValues.iter < 0) {
            DataValues.iter = 0;
        }
        else {
            DataValues.iter = Math.round(value);            
        }
        DataValues.needsUpdate = true; 
    });

    gui.add(DataValues, 'autoPlay');
    gui.add(DataValues, 'playSpeed');//, { Half: 0.5, OneX: 1.0, TwoX: 2.0, FourX: 4.0 } ); //Drop down not working due to event.preventDefault() in OrbitControls on left mouse click
    gui.add(DataValues, 'invert').onChange( function (value) {
        DataValues.needsUpdate = true;    
    });
    gui.add(DataValues, 'autoRotate').onChange( function (value) {
        controls.autoRotate = value;
    });
    gui.add(DataValues, 'rotationSpeed', 0, 100).onChange( function (value) {
        controls.autoRotateSpeed = value;
    });
    gui.add(DataValues, 'next');
    gui.add(DataValues, 'prev');

}

function initScene() {
    //Init timing mechanisms
    clock = new THREE.Clock();
    clock.getDelta();
    time = 0;
    mouse = new THREE.Vector2(0, 0);

    windowHalf = new THREE.Vector2(window.innerWidth / 2, window.innerHeight / 2);
    aspectRatio = window.innerWidth / window.innerHeight;

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(45, aspectRatio, 1, 10000);
    camera.useQuaternion = true;

    camera.position.set(DataValues.zSize, DataValues.xSize*1.5, DataValues.ySize*2);
    camera.lookAt(scene.position);


    // Initialize the renderer
    renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setClearColor(0xdbf7ff);
    renderer.setSize(window.innerWidth, window.innerHeight);

    element = document.getElementById('viewport');
    element.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera);
    controls.target.set(DataValues.xSize/2, DataValues.ySize/2, DataValues.zSize/2);
    controls.autoRotate = DataValues.autoRotate;
    controls.autoRotateSpeed = DataValues.rotationSpeed;
}

function initGrid() {
/*
    var c = new THREE.Clock();
    
    DataValues.mapData.forEach( function (d,i) {
        c.getDelta();       

        if(i < 8) {

        baked[i] = { alive: undefined, dead: undefined };

        //Geometry containers
        var aliveGeo = new THREE.Geometry();
        var deadGeo = new THREE.Geometry();

        //Mesh building blocks
        var geometry = new THREE.CubeGeometry(1, 1, 1);
        var material = DataValues.meshMaterial; 
        

        d.map.forEach( function (d) {
            var mesh = new THREE.Mesh(geometry, undefined);
            //mesh.overdraw = true;
            mesh.position.set(d.x, d.y, d.z);            

            if(d.value) {            
                THREE.GeometryUtils.merge(aliveGeo, mesh);
            } 
            else {
                THREE.GeometryUtils.merge(deadGeo, mesh);
            }
                 
        });

        console.log("[%d]: took %f seconds", i, c.getDelta());
        baked[i].alive = new THREE.Mesh(aliveGeo, DataValues.meshMaterial);
        baked[i].dead = new THREE.Mesh(deadGeo, DataValues.meshMaterial);

        scene.add(baked[i].alive);
        scene.add(baked[i].dead);
        console.log("geo: %d",renderer.info.memory.geometries);
        console.log("calls: %d", renderer.info.render.calls);
        console.log("faces: %d", renderer.info.render.faces);
        console.log("points: %d", renderer.info.render.points);
        console.log("vertices: %d", renderer.info.render.vertices);
        }
    });

    recalcVisible();
*/
    // Old non-merged way

    var x,y;
    for(x=0; x<DataValues.xSize; x++) {
        grid[x] = new Array();
        for(y=0; y<DataValues.ySize; y++) {
            grid[x][y] = new Array();        
        }    
    }

    DataValues.mapData.forEach( function (d) {
        if( d.iteration === DataValues.iter ) {
            var geometry = new THREE.CubeGeometry(1, 1, 1); 
            d.map.forEach( function (d) {
                var material = DataValues.meshMaterial;
                var mesh = new THREE.Mesh(geometry, material);
                mesh.overdraw = true;
                mesh.position.set(d.x, d.y, d.z);
                grid[d.x][d.y][d.z] = mesh;

                mesh.visible = (DataValues.invert) ? !d.value : d.value;

                scene.add(mesh);
            });
        }
    });
    
}

function onResize() {
    windowHalf = new THREE.Vector2(window.innerWidth/2, window.innerHeight/2);
    aspectRatio = window.innerWidth / window.innerHeight;

    camera.aspect = aspectRatio;
    camera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);
}

function recalcVisible() {
    // perform update on baked objects

    baked.forEach( function (d,i) {
        if(i === DataValues.iter) {
            d.alive.visible = !DataValues.invert;
            d.dead.visible = DataValues.invert;
        }
        else {
            d.alive.visible = false;
            d.dead.visible = false;
        }
    });
}

function animate() {
    
    var delta = clock.getDelta();
    time += delta;

    var timeRequired = 1 / DataValues.playSpeed;

    //Check if enough time elapsed since last update
    if( DataValues.autoPlay && time > timeRequired ) {
        console.log("time: %f    needed: %f", time, timeRequired);
        time = 0;
        DataValues.needsUpdate = true;
        DataValues.iter = (DataValues.iter + 1) % DataValues.maxIters;
        //console.log("Dv.iter: %d", DataValues.iter);
    }


    //If updated needed, perform update on mesh grid
    if(DataValues.needsUpdate) {
        var c = new THREE.Clock();
        DataValues.needsUpdate = false;
        /*
        recalcVisible();
        */
        // Old non merged way
        c.getDelta();
        DataValues.mapData[DataValues.iter].map.forEach( function (d) {
            grid[d.x][d.y][d.z].visible = (DataValues.invert) ? !d.value : d.value;
        });
        console.log("gridupdate: took %f seconds", c.getDelta());
        

    }
  
    requestAnimationFrame(animate);
    render();

}

function render() {
    controls.update();
    renderer.render(scene, camera);
    stats.update();
}

window.onload = function () {
	if ( ! Detector.webgl ) Detector.addGetWebGLMessage();
	
	//Load JSON file 16d12b18
    d3.json("data/serial16d12b18.json", function(error, json) {
        if(error) return console.warn(error);

        DataValues.mapData = json.mapData;
        DataValues.xSize = json.xSize;
        DataValues.ySize = json.ySize;
        DataValues.zSize = json.zSize;
        DataValues.maxIters = DataValues.mapData.length;
        DataValues.meshMaterial = new THREE.MeshNormalMaterial({opacity: 0.3, transparent: true});

        /*
        console.log("DataValues:");
        console.log("Size(%d,%d,%d) [%d total]", DataValues.xSize, DataValues.ySize, DataValues.zSize, DataValues.xSize*DataValues.ySize*DataValues.zSize);
        console.log("iter: %d of %d", DataValues.iter, DataValues.maxIters);
        */

        init();
        animate();
    });
};


/*
if ( ! Detector.webgl ) Detector.addGetWebGLMessage();

var container, stats;

var camera, controls, scene, renderer;

init();
render();

function init() {

	camera = new THREE.PerspectiveCamera( 60, window.innerWidth / window.innerHeight, 1, 1000 );
	camera.position.z = 500;

	controls = new THREE.OrbitControls( camera );
	controls.addEventListener( 'change', render );

	scene = new THREE.Scene();
	scene.fog = new THREE.FogExp2( 0xcccccc, 0.002 );

	// world

	var geometry = new THREE.CylinderGeometry( 0, 10, 30, 4, 1 );
	var material =  new THREE.MeshLambertMaterial( { color:0xffffff, shading: THREE.FlatShading } );

	for ( var i = 0; i < 500; i ++ ) {

		var mesh = new THREE.Mesh( geometry, material );
		mesh.position.x = ( Math.random() - 0.5 ) * 1000;
		mesh.position.y = ( Math.random() - 0.5 ) * 1000;
		mesh.position.z = ( Math.random() - 0.5 ) * 1000;
		mesh.updateMatrix();
		mesh.matrixAutoUpdate = false;
		scene.add( mesh );

	}


	// lights

	light = new THREE.DirectionalLight( 0xffffff );
	light.position.set( 1, 1, 1 );
	scene.add( light );

	light = new THREE.DirectionalLight( 0x002288 );
	light.position.set( -1, -1, -1 );
	scene.add( light );

	light = new THREE.AmbientLight( 0x222222 );
	scene.add( light );


	// renderer

	renderer = new THREE.WebGLRenderer( { antialias: false } );
	renderer.setClearColor( scene.fog.color, 1 );
	renderer.setSize( window.innerWidth, window.innerHeight );

	container = document.getElementById( 'container' );
	container.appendChild( renderer.domElement );

	stats = new Stats();
	stats.domElement.style.position = 'absolute';
	stats.domElement.style.top = '0px';
	stats.domElement.style.zIndex = 100;
	container.appendChild( stats.domElement );

	//

	window.addEventListener( 'resize', onWindowResize, false );

}

function onWindowResize() {

	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();

	renderer.setSize( window.innerWidth, window.innerHeight );

	render();

}

function render() {
	renderer.render( scene, camera );
	stats.update();
}
*/
