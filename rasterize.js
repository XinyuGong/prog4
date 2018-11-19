/* GLOBAL CONSTANTS AND VARIABLES */

/* assignment specific globals */
const URL = 'https://ncsucgclass.github.io/prog4/';
const WIN_Z = 0;  // default graphics window z coord in world space
const WIN_LEFT = 0; const WIN_RIGHT = 1;  // default left and right x coords in world space
const WIN_BOTTOM = 0; const WIN_TOP = 1;  // default top and bottom y coords in world space
const INPUT_TRIANGLES_URL = "https://ncsucgclass.github.io/prog4/triangles.json"; // triangles file loc
var eye = new vec3.fromValues(0.5, 0.5, -0.5); // default eye position in world space
var lookAt = new vec3.fromValues(0.0, 0.0, 1.0);
var lookUp = new vec3.fromValues(0.0, 1.0, 0.0);
var lightPos = new vec3.fromValues(-3, 1, -0.5);

/* webgl globals */
var gl = null; // the all powerful gl object. It's all here folks!
var vertexPositionAttrib; // where to put position for vertex shader
var vertexUVAttrib;
var mAmbientLoc;
var mDiffuseLoc;
var mSpecularLoc;
var mWeightsLoc;
var mAlphaLoc;
var vertexNormalAttrib;
var lightLoc;
var eyeLoc;
var projectionLoc;
var transLoc;
var rotatLoc;
var scaleLoc;
var modeLoc;
var nLoc;
var viewMatLoc;
var transMatLoc;
var isHighlighting = false;
var inHighlight = -1;
var models = [];
var loadedImgs = 0;

class Model {
    constructor(inputTriangle) {
        this.triBufferSize = 0;
        this.transMat = new mat4.create();
        this.rotatMat = new mat4.create();
        this.scaleMat = new mat4.create();
        this.mode = 1; // 0 - phong; 1 - blinn phong
        if (inputTriangle != String.null) {
            this.center = getCenter(inputTriangle.vertices);
            this.n = inputTriangle.material.n;
            var whichSetVert; // index of vertex in current triangle set
            var whichSetTri; // index of triangle in current triangle set
            var coordArray = []; // 1D array of vertex coords for WebGL
            var indexArray = []; // 1D array of vertex indices for WebGL
            var uvArray = []; // Texture coords
            var normalArray = [];
            var vtxBufferSize = 0; // the number of vertices in the vertex buffer
            var vtxToAdd = []; // vtx coords to add to the coord array
            var nrmToAdd = [];
            var uvToAdd = [];
    
            // set up the vertex coord array
            for (whichSetVert=0; whichSetVert<inputTriangle.vertices.length; whichSetVert++) {
                vtxToAdd = inputTriangle.vertices[whichSetVert];
                nrmToAdd = inputTriangle.normals[whichSetVert];
                uvToAdd = inputTriangle.uvs[whichSetVert];
                coordArray.push(vtxToAdd[0],vtxToAdd[1],vtxToAdd[2]);
                normalArray.push(nrmToAdd[0], nrmToAdd[1], nrmToAdd[2]);
                uvArray.push(uvToAdd[0], uvToAdd[1]);
            } // end for vertices in set

            // read colors
            this.ambient = new vec3.fromValues(
                inputTriangle.material.ambient[0],
                inputTriangle.material.ambient[1],
                inputTriangle.material.ambient[2]
            );
            this.diffuse = new vec3.fromValues(
                inputTriangle.material.diffuse[0],
                inputTriangle.material.diffuse[1],
                inputTriangle.material.diffuse[2]
            );
            this.specular = new vec3.fromValues(
                inputTriangle.material.specular[0],
                inputTriangle.material.specular[1],
                inputTriangle.material.specular[2]
            );
            this.weights = new vec3.fromValues(1, 1, 1);
            this.alpha = inputTriangle.material.alpha;

            // read texture
            this.texturePath = URL + inputTriangle.material.texture;
            this.image = new Image();
            this.imageLoad = false;
            
            // set up the triangle index array, adjusting indices across sets
            for (whichSetTri=0; whichSetTri<inputTriangle.triangles.length; whichSetTri++) {
                indexArray.push(
                    inputTriangle.triangles[whichSetTri][0],
                    inputTriangle.triangles[whichSetTri][1],
                    inputTriangle.triangles[whichSetTri][2]
                );
            } // end for triangles in set

            vtxBufferSize += inputTriangle.vertices.length; // total number of vertices
            this.triBufferSize += inputTriangle.triangles.length; // total number of tris

            this.triBufferSize *= 3; // now total number of indices
    
            // send the vertex coords to webGL
            this.vertexBuffer = gl.createBuffer(); // init empty vertex coord buffer
            gl.bindBuffer(gl.ARRAY_BUFFER,this.vertexBuffer); // activate that buffer
            gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(coordArray),gl.STATIC_DRAW); // coords to that buffer
    
            // normal
            this.normalBuffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, this.normalBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normalArray), gl.STATIC_DRAW);

            // texture coords
            this.uvBuffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(uvArray), gl.STATIC_DRAW);
            
            // send the triangle indices to webGL
            this.triangleBuffer = gl.createBuffer(); // init empty triangle index buffer
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.triangleBuffer); // activate that buffer
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(indexArray),gl.STATIC_DRAW); // indices to that buffer
    
        } // end if triangles found
    }
}

// ASSIGNMENT HELPER FUNCTIONS

// get the JSON file from the passed URL
function getJSONFile(url,descr) {
    try {
        if ((typeof(url) !== "string") || (typeof(descr) !== "string"))
            throw "getJSONFile: parameter not a string";
        else {
            var httpReq = new XMLHttpRequest(); // a new http request
            httpReq.open("GET",url,false); // init the request
            httpReq.send(null); // send the request
            var startTime = Date.now();
            while ((httpReq.status !== 200) && (httpReq.readyState !== XMLHttpRequest.DONE)) {
                if ((Date.now()-startTime) > 3000)
                    break;
            } // until its loaded or we time out after three seconds
            if ((httpReq.status !== 200) || (httpReq.readyState !== XMLHttpRequest.DONE))
                throw "Unable to open "+descr+" file!";
            else
                return JSON.parse(httpReq.response); 
        } // end if good params
    } // end try    
    
    catch(e) {
        console.log(e);
        return(String.null);
    }
} // end get input spheres

// set up the webGL environment
function setupWebGL() {

    // Get the canvas and context
    var canvas = document.getElementById("myWebGLCanvas"); // create a js canvas
    gl = canvas.getContext("webgl"); // get a webgl object from it
    
    try {
      if (gl == null) {
        throw "unable to create gl context -- is your browser gl ready?";
      } else {
        gl.clearColor(0.0, 0.0, 0.0, 1.0); // use black when we clear the frame buffer
        gl.clearDepth(1.0); // use max when we clear the depth buffer
        gl.enable(gl.DEPTH_TEST); // use hidden surface removal (with zbuffering)
      }
    } // end try
    
    catch(e) {
      console.log(e);
    } // end catch
 
} // end setupWebGL

// read triangles in, load them into webgl buffers
function loadTriangles() {
    var inputTriangles = getJSONFile(INPUT_TRIANGLES_URL,"triangles");
    for (var i = 0; i < inputTriangles.length; i ++) {
        models.push(new Model(inputTriangles[i]));
    }
} // end load triangles

// setup the webGL shaders
function setupShaders() {
    
    // define fragment shader in essl using es6 template strings
    var fShaderCode = `
        precision mediump float;

        varying vec3 normalInterp;
        varying vec3 vertPos;
        varying vec2 UV;

        uniform vec3 lightPos, eye;
        uniform vec3 mAmbient, mDiffuse, mSpecular, mWeights;
        uniform float n, mAlpha;
        uniform int mode; // mode: 0 - replace; 1 - modulate
        uniform sampler2D texture;

        void main(void) {
            vec3 normal = normalize(normalInterp);
            vec3 lightDir = normalize(lightPos - vertPos);
            
            float lambertian = max(dot(lightDir, normal), 0.0);
            float specular = 0.0;

            if (lambertian > 0.0) {
                vec3 viewDir = normalize(eye - vertPos);
                vec3 halfDir = normalize(lightDir + viewDir);
                float specAngle = max(dot(halfDir, normal), 0.0);
                specular = pow(specAngle, n);
            }
            //gl_FragColor = vec4(mWeights[0] * mAmbient + mWeights[1] * lambertian * mDiffuse + mWeights[2] * specular * mSpecular, 1.0);
            vec4 textureColor = texture2D(texture, UV);
            vec3 cAmbient;
            vec3 cDiffuse;
            vec3 cSpecular;
            if (mode == 0) {
                // replace
                cAmbient = vec3(1.0, 1.0, 1.0);
                cDiffuse = vec3(1.0, 1.0, 1.0);
                cSpecular = vec3(1.0, 1.0, 1.0);
            } else {
                // modulate
                cAmbient = mAmbient;
                cDiffuse = mDiffuse;
                cSpecular = mSpecular;
            }
            gl_FragColor = vec4(textureColor.rgb * (cAmbient + lambertian * cDiffuse + specular * cSpecular), textureColor.a * mAlpha);
        }
    `;
    
    // define vertex shader in essl using es6 template strings
    var vShaderCode = `
        attribute vec3 vertexPosition;
        attribute vec3 vertexNormal;
        attribute vec2 vertexUV;

        uniform mat4 projection, viewMatrix, transMatrix;

        varying vec3 normalInterp;
        varying vec3 vertPos;
        varying vec2 UV;

        void main(void) {
            gl_Position = projection * viewMatrix * (transMatrix * vec4(vertexPosition, 1.0));
            vertPos = vec3(transMatrix * vec4(vertexPosition, 1.0));
            normalInterp = vec3(normalize(transMatrix * vec4(vertexNormal, 0.0)));
            UV = vertexUV;
        }
    `;
    
    try {
        // console.log("fragment shader: "+fShaderCode);
        var fShader = gl.createShader(gl.FRAGMENT_SHADER); // create frag shader
        gl.shaderSource(fShader,fShaderCode); // attach code to shader
        gl.compileShader(fShader); // compile the code for gpu execution

        // console.log("vertex shader: "+vShaderCode);
        var vShader = gl.createShader(gl.VERTEX_SHADER); // create vertex shader
        gl.shaderSource(vShader,vShaderCode); // attach code to shader
        gl.compileShader(vShader); // compile the code for gpu execution
            
        if (!gl.getShaderParameter(fShader, gl.COMPILE_STATUS)) { // bad frag shader compile
            throw "error during fragment shader compile: " + gl.getShaderInfoLog(fShader);  
            gl.deleteShader(fShader);
        } else if (!gl.getShaderParameter(vShader, gl.COMPILE_STATUS)) { // bad vertex shader compile
            throw "error during vertex shader compile: " + gl.getShaderInfoLog(vShader);  
            gl.deleteShader(vShader);
        } else { // no compile errors
            gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
            gl.enable(gl.DEPTH_TEST);
            var shaderProgram = gl.createProgram(); // create the single shader program
            gl.attachShader(shaderProgram, fShader); // put frag shader in program
            gl.attachShader(shaderProgram, vShader); // put vertex shader in program
            gl.linkProgram(shaderProgram); // link program into gl context

            if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) { // bad program link
                throw "error during shader program linking: " + gl.getProgramInfoLog(shaderProgram);
            } else { // no shader program link errors
                gl.useProgram(shaderProgram); // activate shader program (frag and vert)
                vertexPositionAttrib = // get pointer to vertex shader input
                    gl.getAttribLocation(shaderProgram, "vertexPosition");
                gl.enableVertexAttribArray(vertexPositionAttrib); // input to shader from array
                
                // projection and eye
                projectionLoc = gl.getUniformLocation(shaderProgram, 'projection');
                eyeLoc = gl.getUniformLocation(shaderProgram, 'eye');
                viewMatLoc = gl.getUniformLocation(shaderProgram, 'viewMatrix');

                // color attributes
                mAmbientLoc = gl.getUniformLocation(shaderProgram, 'mAmbient');
                mDiffuseLoc = gl.getUniformLocation(shaderProgram, 'mDiffuse');
                mSpecularLoc = gl.getUniformLocation(shaderProgram, 'mSpecular');
                mWeightsLoc = gl.getUniformLocation(shaderProgram, 'mWeights');
                mAlphaLoc = gl.getUniformLocation(shaderProgram, 'mAlpha');
                vertexUVAttrib = gl.getAttribLocation(shaderProgram, 'vertexUV');
                gl.enableVertexAttribArray(vertexUVAttrib);

                // mode
                modeLoc = gl.getUniformLocation(shaderProgram, 'mode');

                // n
                nLoc = gl.getUniformLocation(shaderProgram, 'n');

                // normal attributes
                vertexNormalAttrib = gl.getAttribLocation(shaderProgram, 'vertexNormal');
                gl.enableVertexAttribArray(vertexNormalAttrib);

                // transformation
                transMatLoc = gl.getUniformLocation(shaderProgram, 'transMatrix');

                // light
                lightLoc = gl.getUniformLocation(shaderProgram, 'lightPos');
            } // end if no shader program link errors
        } // end if no compile errors
    } // end try 
    
    catch(e) {
        console.log(e);
    } // end catch
} // end setup shaders

function loadTexture(modelIndex) {
    models[modelIndex].image.crossOrigin = 'Anonymous';
    models[modelIndex].image.src = models[modelIndex].texturePath;
    models[modelIndex].image.addEventListener('load', function() {
        models[modelIndex].imageLoad = true;
        renderTriangles();
    });
}

// render the loaded model
function renderTriangles() {
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); // clear frame/depth buffers
    // Enable alpha blending and set the percentage blending factors
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    //requestAnimationFrame(renderTriangles);
    for (var i = 0; i < models.length; i ++) {
        var model = models[i];
        // vertex buffer: activate and feed into vertex shader
        gl.bindBuffer(gl.ARRAY_BUFFER,model.vertexBuffer); // activate
        gl.vertexAttribPointer(vertexPositionAttrib,3,gl.FLOAT,false,0,0); // feed

        // projection and eye
        var projectionMat = new mat4.create();
        var viewMat = new mat4.create();
        var focal = new vec3.create();
        vec3.add(focal, eye, lookAt);mat4.lookAt(viewMat, eye, focal, lookUp);
        gl.uniformMatrix4fv(viewMatLoc, false, viewMat);
        mat4.perspective(projectionMat, Math.PI/2, 1, 0.5, 2);
        gl.uniformMatrix4fv(projectionLoc, false, projectionMat);
        gl.uniform3fv(eyeLoc, eye);

        // color
        gl.uniform3fv(mAmbientLoc, model.ambient);
        gl.uniform3fv(mDiffuseLoc, model.diffuse);
        gl.uniform3fv(mSpecularLoc, model.specular);
        gl.uniform3fv(mWeightsLoc, model.weights);
        gl.uniform1f(mAlphaLoc, model.alpha);
        gl.bindBuffer(gl.ARRAY_BUFFER, model.uvBuffer);
        gl.vertexAttribPointer(vertexUVAttrib, 2, gl.FLOAT, false, 0, 0);
        var texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        if (model.imageLoad) {
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA,gl.UNSIGNED_BYTE, model.image);
            gl.generateMipmap(gl.TEXTURE_2D);
        } else {
            // use dummy texture
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 255, 255]));
        }

        // mode
        gl.uniform1i(modeLoc, model.mode);

        // normal
        gl.bindBuffer(gl.ARRAY_BUFFER, model.normalBuffer);
        gl.vertexAttribPointer(vertexNormalAttrib, 3, gl.FLOAT, false, 0, 0);

        // n
        gl.uniform1f(nLoc, model.n);

        // transformation
        var transMat = mat4.create();
        mat4.mul(transMat, model.scaleMat, transMat);
        mat4.mul(transMat, model.rotatMat, transMat);
        mat4.mul(transMat, model.transMat, transMat);
        gl.uniformMatrix4fv(transMatLoc, false, transMat);

        // light
        gl.uniform3fv(lightLoc, lightPos);

        // triangle buffer: activate and render
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,model.triangleBuffer); // activate
        gl.drawElements(gl.TRIANGLES,model.triBufferSize,gl.UNSIGNED_SHORT,0); // render
    }
} // end render triangles

function  mat4LookAt(viewMatrix) {
    var right = new vec3.create();
    vec3.cross(right, lookAt, lookUp);
    vec3.normalize(right, right);

    mat4.set(
        viewMatrix,
        right[0], lookUp[0], -lookAt[0], 0.0,
        right[1], lookUp[1], -lookAt[1], 0.0,
        right[2], lookUp[2], -lookAt[2], 0.0,
        -vec3.dot(right, eye), -vec3.dot(lookUp, eye), vec3.dot(lookAt, eye), 1.0
    );
}

function mat4Perspective(a, fov, aspect, zNear, zFar) {
    var f = 1.0 / Math.tan (fov/2.0 * (Math.PI / 180.0));
    a[0] = f / aspect;
    a[1 * 4 + 1] = f;
    a[2 * 4 + 2] = (zFar + zNear)  / (zNear - zFar);
    a[3 * 4 + 2] = (2.0 * zFar * zNear) / (zNear - zFar);
    a[2 * 4 + 3] = -1.0;
    a[3 * 4 + 3] = 0.0;
}

// keyboard handler
function keyDownHandler(event) {
    switch(event.key) {
        case 'a': transX(-0.1); break; // translate view left
        case 'd': transX(0.1); break; // translate view right
        case 'q': transY(0.1); break; // translate view up
        case 'e': transY(-0.1); break; // translate view down
        case 'w': transZ(0.1); break; // translate view forward
        case 's': transZ(-0.1); break; // translate view backward
        case 'A': rotateY(5); break; // rotate view left
        case 'D': rotateY(-5); break; // rotate view right
        case 'W': rotateX(5); break; // rotate view up
        case 'S': rotateX(-5); break; // rotate view down
        case 'ArrowLeft': highlightOn(-1); break; // select previous model
        case 'ArrowRight': highlightOn(1); break; // select next model
        case ' ': highlightOff(); break; // cancel selection
    }

    // operations only valid if selecting a model
    if (isHighlighting) {
        var model = models[inHighlight];
        switch(event.key) {
            case 'b': model.mode = 1 - model.mode; break; // toggle lighting mode
            case 'n': model.n = (model.n + 1) % 21; break; // increase n
            case '1': model.weights[0] += 0.1; if (model.weights[0] > 1) model.weights[0] = 0; break;// increase the ambient weight
            case '2': model.weights[1] += 0.1; if (model.weights[1] > 1) model.weights[1] = 0; break;// increase the ambient weight
            case '3': model.weights[2] += 0.1; if (model.weights[2] > 1) model.weights[2] = 0; break;// increase the ambient weight
            case 'k': mTransX(-0.1); break; // translate model left
            case ';': mTransX(0.1); break; // translate model right
            case 'o': mTransZ(0.1); break; // translate model forward
            case 'l': mTransZ(-0.1); break; // translate model backward
            case 'i': mTransY(0.1); break; // translate model up
            case 'p': mTransY(-0.1); break; // translate model down
            case 'K': mRotateY(5); break; // rotate model left
            case ':': mRotateY(-5); break; // rotate model right
            case 'O': mRotateX(-5); break; // rotate model forward
            case 'L': mRotateX(5); break; // rotate model backward
            case 'I': mRotateZ(5); break; // rotate model clockwise
            case 'P': mRotateZ(-5); break; // rotate model counterclockwise
        }
    }
    renderTriangles();
}

// view translation
function transX(amount) { // move right if amount > 0
    var right = new vec3.create();
    vec3.cross(right, lookAt, lookUp);
    vec3.normalize(right, right);
    
    var trans = new vec3.create();
    vec3.scale(trans, right, amount);
    vec3.add(eye, eye, trans);
}

function transY(amount) { // move up if amount > 0
    var trans = new vec3.create();
    vec3.scale(trans, lookUp, amount);
    vec3.add(eye, eye, trans);
}

function transZ(amount) { // move forward if amount > 0
    var trans = new vec3.create();
    vec3.scale(trans, lookAt, amount);
    vec3.add(eye, eye, trans);
}

// model translation
function mTransX(amount) { // move right if amount > 0
    var right = new vec3.create();
    vec3.cross(right, lookAt, lookUp);
    vec3.normalize(right, right);

    var trans = new vec3.create();
    vec3.scale(trans, right, amount);
    models[inHighlight].transMat[12] += trans[0];
    models[inHighlight].transMat[13] += trans[1];
    models[inHighlight].transMat[14] += trans[2];
}

function mTransY(amount) { // move up if amount > 0
    var trans = new vec3.create();
    vec3.scale(trans, lookUp, amount);
    models[inHighlight].transMat[12] += trans[0];
    models[inHighlight].transMat[13] += trans[1];
    models[inHighlight].transMat[14] += trans[2];
}

function mTransZ(amount) { // move forward if amount > 0
    var trans = new vec3.create();
    vec3.scale(trans, lookAt, amount);
    models[inHighlight].transMat[12] += trans[0];
    models[inHighlight].transMat[13] += trans[1];
    models[inHighlight].transMat[14] += trans[2];
}

// view rotation, angle in degrees
function rotateY(deg) { // turn left if deg > 0
    var rotMat = new mat4.create();
    mat4.fromRotation(rotMat, deg / 180 * Math.PI, lookUp);   
    vec3.transformMat4(lookAt, lookAt, rotMat);
}

function rotateX(deg) { // turn up if deg > 0
    var right = new vec3.create();
    vec3.cross(right, lookAt, lookUp);
    vec3.normalize(right, right);

    var rotMat = new mat4.create();
    mat4.fromRotation(rotMat, deg / 180 * Math.PI, right);   
    vec3.transformMat4(lookAt, lookAt, rotMat);
    vec3.transformMat4(lookUp, lookUp, rotMat);
}

// model rotation, angle in degrees
function mRotateY(deg) { // turn left if deg > 0
    var model = models[inHighlight];
    var rotMat = new mat4.create();
    mat4.fromRotation(rotMat, deg / 180 * Math.PI, lookUp);   
    mat4.mul(model.rotatMat, model.rotatMat, rotMat);
    rotMat = model.rotatMat;
    rotMat[12] = vec3.dot(new vec3.fromValues(1-rotMat[0], -rotMat[4], -rotMat[8]), model.center);
    rotMat[13] = vec3.dot(new vec3.fromValues(-rotMat[1], 1-rotMat[5], -rotMat[9]), model.center);
    rotMat[14] = vec3.dot(new vec3.fromValues(-rotMat[2], -rotMat[6], 1-rotMat[10]), model.center);
}

function mRotateX(deg) { // turn up if deg > 0
    var right = new vec3.create();
    vec3.cross(right, lookAt, lookUp);
    vec3.normalize(right, right);
    var model = models[inHighlight];
    var rotMat = new mat4.create();
    mat4.fromRotation(rotMat, deg / 180 * Math.PI, right);   
    mat4.mul(model.rotatMat, model.rotatMat, rotMat);
    rotMat = model.rotatMat;
    rotMat[12] = vec3.dot(new vec3.fromValues(1-rotMat[0], -rotMat[4], -rotMat[8]), model.center);
    rotMat[13] = vec3.dot(new vec3.fromValues(-rotMat[1], 1-rotMat[5], -rotMat[9]), model.center);
    rotMat[14] = vec3.dot(new vec3.fromValues(-rotMat[2], -rotMat[6], 1-rotMat[10]), model.center);
}

function mRotateZ(deg) { // turn clockwise if deg > 0
    var model = models[inHighlight];
    var rotMat = new mat4.create();
    mat4.fromRotation(rotMat, deg / 180 * Math.PI, lookAt);
    mat4.mul(model.rotatMat, model.rotatMat, rotMat);
    rotMat = model.rotatMat;
    rotMat[12] = vec3.dot(new vec3.fromValues(1-rotMat[0], -rotMat[4], -rotMat[8]), model.center);
    rotMat[13] = vec3.dot(new vec3.fromValues(-rotMat[1], 1-rotMat[5], -rotMat[9]), model.center);
    rotMat[14] = vec3.dot(new vec3.fromValues(-rotMat[2], -rotMat[6], 1-rotMat[10]), model.center);
}

// highlight
function highlightOn(next) { // highlight next one if next = 1, otherwise the previous one
    if (isHighlighting) {
        highlightOff();
        highlightOn(next);
    } else {
        isHighlighting = true;
        inHighlight += next;
        if (inHighlight < 0) inHighlight = models.length - 1;
        if (inHighlight >= models.length) inHighlight = 0;
        var model = models[inHighlight];
        mat4.set(
            model.scaleMat,
            1.2, 0, 0, 0,
            0, 1.2, 0, 0,
            0, 0, 1.2, 0,
            -model.center[0]/5, -model.center[1]/5, -model.center[2]/5, 1
        );
    }
}

function highlightOff() {
    if (isHighlighting) {
        isHighlighting = false;
        mat4.identity(models[inHighlight].scaleMat);
    }
}

function getCenter(vertices) {
    var coord = [[], [], []];
    for (var i = 0; i < 3; i ++) {
        for (var j = 0; j < vertices.length; j ++) {
            coord[i].push(vertices[j][i]);
        }
    }
    var center = new vec3.fromValues(
        (Math.max(...coord[0]) + Math.min(...coord[0])) / 2,
        (Math.max(...coord[1]) + Math.min(...coord[1])) / 2,
        (Math.max(...coord[2]) + Math.min(...coord[2])) / 2
    );
    return center;
}

/* MAIN -- HERE is where execution begins after window load */

function main() {

    setupWebGL(); // set up the webGL environment
    loadTriangles(); // load in the triangles from tri file
    setupShaders(); // setup the webGL shaders
    for (var i = 0; i < models.length; i ++) {
        loadTexture(i);
    }
    renderTriangles(); // draw the triangles using webGL

    document.addEventListener('keydown', keyDownHandler, false);

} // end main
