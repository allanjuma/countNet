
var svg;
   // Bind data
   
   function setRect(data,baseRect){
    baseRect = baseRect.data(data);

    // set the rects
    baseRect.transition()
            .duration(200)
            .attr('width', xr)
            .attr('height', xr)
            .attr('x', randomPosition)
            .attr('y', randomPosition)
            .style('fill', tcColours[randomTcColour()]);

    baseRect.enter()
            .append('rect')
            .attr('width', xr)
            .attr('height', xr)
            .attr('x', randomPosition)
            .attr('y', randomPosition)
            .style('fill', tcColours[randomTcColour()]);


     
   }
  
  function setCirc(data, baseCircle){
  	 baseCircle = baseCircle.data(data);
    
    // set the circles
    baseCircle.transition()
            .duration(250)
            .attr('r', xr)
            .attr('cx', randomPosition)
            .attr('cy', randomPosition)
            .attr('fill', "none")
            .attr("stroke-width", 4)
            .style('stroke', tcColours[randomTcColour()]);
 
    baseCircle.enter()
            .append('circle')
            .attr('r', xr)
            .attr('cx', randomPosition)
            .attr('cy', randomPosition)
            .attr('fill', "none")
            .attr("stroke-width", 4)
            .style('stroke', tcColours[randomTcColour()]);
  }

async function update(shape,count) {
  svg.selectAll("*").remove();
    

// create data
var data = [];
for (var i=0; i < count; i++) {
 data.push(i);
}
// create random data
var ranData = [];
for (var i=0; i < getRandomInt(1, 30); i++) {
 ranData.push(i);
}
console.log(count,data);

  

  if(count>1){
  
  if(shape==1){

setRect(data,svg.selectAll('rect')) 
setCirc(ranData,svg.selectAll('circle'))
  }	else{
  	
setRect(ranData,svg.selectAll('rect')) 
setCirc(data,svg.selectAll('circle'))
  }
  } else{

if(shape==1){
 setRect(data,svg.selectAll('rect'))   
  }else if(shape==2){
setCirc(data,svg.selectAll('circle'))
  }
  }
  
return true;

}

function datasetManager(){
// Scale for radius
xr = d3.scale
        .linear()
        .domain([10, 30])
        .range([30, 50]);

// Scale for random position
randomPosition = function(d) {
    return Math.random() * 200;
}


tcColours = ['#FDBB30', '#EE3124', '#EC008C', '#F47521', '#7AC143', '#00B0DD'];
randomTcColour = function() {
  return Math.floor(Math.random() * tcColours.length);
};


// SVG viewport
svg = d3.select('.viewGen')
  .append('svg')
    .attr('width', 224)
    .attr('height', 224);

}


function initDBstore(sn) {
	  var indexedDBOpenRequest;

    indexedDBOpenRequest = indexedDB.open(sn);
  // This top-level error handler will be invoked any time there's an IndexedDB-related error.
  indexedDBOpenRequest.onerror = function(error) {
      
    console.error('IndexedDB error:', error);
     // indexedDB.createObjectStore(STORE_NAME);
  };

  // This should only execute if there's a need to create a new database for the given IDB_VERSION.
  indexedDBOpenRequest.onupgradeneeded = function() {
  //  this.result.createObjectStore(STORE_NAME, {keyPath: 'url'});
      if(sn=='offana'){
          this.result.createObjectStore(sn, {keyPath: 'url'});
      }else{
       
    this.result.createObjectStore(sn);   
      }
  };

  // This will execute each time the database is opened.
  indexedDBOpenRequest.onsuccess = function() {
      if(sn=='data'){idbDatabase = this.result;}else if(sn=='notes'){notesDatabase = this.result;}else if(sn=='offana'){anaDatabase = this.result;}
   //
	  indexDbReady = new Promise((resolve, reject) => {
  	resolve('ready');
}); 
  };
}
  
// Helper method to get the object store that we care about.
function getAnaStore(storeName) {
    
  return anaDatabase.transaction('offana', 'readwrite').objectStore(storeName);
    
        
}

function getNotesStore(storeName) {
    
  return notesDatabase.transaction('notes', 'readwrite').objectStore(storeName);
    
        
}


function getObjectStore(storeName) {
    
    
  return idbDatabase.transaction('data', 'readwrite').objectStore(storeName);
    
        
}



function DBstore(storeName, data) {
  getObjectStore(storeName, 'readwrite').add({
    url: data
  });
}


	
var idbDatabase;
var IDB_VERSION = 1;
 //var indexedDB = window.indexedDB || window.webkitIndexedDB || window.mozIndexedDB || window.OIndexedDB || window.msIndexedDB,
    //    IDBTransaction = window.IDBTransaction || window.webkitIDBTransaction || window.OIDBTransaction || window.msIDBTransaction;

    	initDBstore('data');
indexDbReady = Promise.race([]);


//end indexedDB store manager

const CANVAS_WIDTH = 224
const CANVAS_HEIGHT = 224 


//should not be hardcoded!
var NUM_CLASSES =2;
var class_names = {}





 htmlToElement = function(html) {
    var template = document.createElement('template');
    html = html.trim(); // Never return a text node of whitespace as the result
    template.innerHTML = html;
    return template.content.firstChild;
}
function startWebcam(video) {
	navigator.getUserMedia  = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
	var facingMode = "environment";
	var constraints = {
	  audio: false,
	  video: {
	    facingMode: facingMode
	  }
	}
	navigator.mediaDevices.getUserMedia(constraints).then(function success(stream) {
		video.srcObject = stream;
		//adjustVideoSize(video.width, video.height);
	});

	video.addEventListener('click', function() {
		if (facingMode == "user") {
			facingMode = "environment";
		} else {
			facingMode = "user";
		} 

		constraints = {
			audio: false,
			video: {
		  	facingMode: facingMode
			}
		}  

		navigator.mediaDevices.getUserMedia(constraints).then(function success(stream) {
		  	video.srcObject = stream;	
		  	//adjustVideoSize(video.width, video.height);
		});
	});
	return video;
}

let ACTIVATION; //for console testing

function getCanvasFrame(som){
  if(som=='single'){

  var canvas = document.querySelector('.singleCanvas');
   
  }else if (som=='many'){

   var canvas = document.querySelector('.manyCanvas');

  }
 
  return canvas;

}


 getVideoFrame = function(som){
  document.querySelector('input').value='';
  if(som=='single'){

  var canvas = document.querySelector('.singleCanvas');
   
  }else if (som=='many'){

   var canvas = document.querySelector('.manyCanvas');

  }
  
  var ctx = canvas.getContext('2d');

  // Change the size here
  canvas.width = CANVAS_WIDTH;
  canvas.height =  CANVAS_HEIGHT;
  ctx.drawImage(document.querySelector('video'), 0, 0, canvas.width, canvas.height);
 

}
/**
 * A class that wraps webcam video elements to capture Tensor4Ds.
 */
 class Webcam {
  /**
   * @param {HTMLVideoElement} webcamElement A HTMLVideoElement representing the webcam feed.
   */
  constructor(webcamElement) {
    this.webcamElement = webcamElement;
  }

  /**
   * Captures a frame from the webcam and normalizes it between -1 and 1.
   * Returns a batched image (1-element batch) of shape [1, w, h, c].
   */
  capture() {
    return tf.tidy(() => {
      // Reads the image as a Tensor from the webcam <video> element.
      const webcamImage = tf.fromPixels(this.webcamElement);

      // Crop the image so we're using the center square of the rectangular
      // webcam.
      const croppedImage = this.cropImage(webcamImage);

      // Expand the outer most dimension so we have a batch size of 1.
      const batchedImage = croppedImage.expandDims(0);

      // Normalize the image between -1 and 1. The image comes in between 0-255,
      // so we divide by 127 and subtract 1.
      return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
  }

  /**
   * Crops an image tensor so we get a square image with no white space.
   * @param {Tensor4D} img An input image Tensor to crop.
   */
  cropImage(img) {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [224, 224, 3]);
  }
}


/**
 * A dataset for webcam controls which allows the user to add example Tensors
 * for particular labels. This object will concat them into two large xs and ys.
 */
 class ModelDataset {
  
  /**
   * Adds an example to the controller dataset.
   * @param {Tensor} example A tensor representing the example. It can be an image,
   *     an activation, or any other type of Tensor.
   * @param {number} label The label of the example. Should be an umber.
   */
 addExample(example, label) {
 console.log(example, label) 
    // One-hot encode the label.
    
const y = tf.tidy(() => tf.oneHot(tf.tensor1d([label]).toInt(), 10));


    if (this.xs == null) {
      // For the first example that gets added, keep example and y so that the
      // modelDataset owns the memory of the inputs. This makes sure that
      // if addExample() is called in a tf.tidy(), these Tensors will not get
      // disposed.
      this.xs = tf.keep(example);
      this.ys = tf.keep(y);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));

      const oldY = this.ys;
      this.ys = tf.keep(oldY.concat(y, 0));

      oldX.dispose();
      oldY.dispose();
      y.dispose();
    }
  }
}

function delay(){
	return new Promise(function(resolve,reject){setTimeout(resolve, 350)})
}

async function createTestData(shape,count){
      document.querySelector('.singleCanvas').getContext("2d").clearRect(0, 0, document.querySelector('.singleCanvas').width, document.querySelector('.singleCanvas').height);
        document.querySelector('.manyCanvas').getContext("2d").clearRect(0, 0, document.querySelector('.manyCanvas').width, document.querySelector('.manyCanvas').height);
    
  document.querySelector('.testCount').value=count;
return update(shape,count).then(function(e){

    // this is just a JavaScript (HTML) image
    var img = new Image();
    // http://en.wikipedia.org/wiki/SVG#Native_support
    // https://developer.mozilla.org/en/DOM/window.btoa
    
    img.onload = function() {
        // after this, Canvas’ origin-clean is DIRTY
document.querySelector('.manyCanvas').getContext("2d").drawImage(img, 0, 0);
      return update(shape,1).then(function(e){

   
    // this is just a JavaScript (HTML) image
    var imgi = new Image();
    // http://en.wikipedia.org/wiki/SVG#Native_support
    // https://developer.mozilla.org/en/DOM/window.btoa
    
    imgi.onload = function() {
        // after this, Canvas’ origin-clean is DIRTY
        document.querySelector('.singleCanvas').getContext("2d").drawImage(imgi, 0, 0);
        setExampleHandler().then(function(e){
        return;	
        });
        
    }
    imgi.src = "data:image/svg+xml;base64," + btoa((new XMLSerializer()).serializeToString(document.querySelector('svg')));

  })
    }
    
img.src = "data:image/svg+xml;base64," + btoa((new XMLSerializer()).serializeToString(document.querySelector('svg')));

  })
 

    
}

// The dataset object where we will store activations.
var modelDataset = new ModelDataset();

let mobilenet
let countnet
let model;


const MODEL_FILE_URL = 'tensorflowjs_model.pb';
const WEIGHT_MANIFEST_FILE_URL = 'weights_manifest.json';

class MobileNet {
  constructor() {}

  async load() {
    const tModel = await tf.loadFrozenModel(
        MODEL_FILE_URL,
        WEIGHT_MANIFEST_FILE_URL);
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }
 }


 

// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function loadMobilenet() {
  const mnet = await tf.loadModel('mobilenet_v1_0.25_224/model.json');

    // new MobileNet().load().then(function(e){console.log(e)});

  // Return a model that outputs an internal activation.
  const layer = mnet.getLayer('conv_pw_13_relu');
  // for(let i = 0;i<mobilenet.layers.length;i++){
  //   console.log(mobilenet.layers[i].name,mobilenet.layers[i].output.shape);
  // }

  return tf.model({inputs: mnet.inputs, outputs: layer.output});
}


// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.

async function loadCountNet() {
	try{
		idbModel = await tf.loadModel('indexeddb://countnet');
		console.log('loaded model from db');
		return idbModel;
	}catch(err){

		console.log(err+' count not load model from indexdb');
	
  var mobilenet = await loadMobilenet();

  // Return a model that outputs an internal activation.
 // const layer = mobilenet.getLayer('conv_pw_13_relu');
  // for(let i = 0;i<mobilenet.layers.length;i++){
  //   console.log(mobilenet.layers[i].name,mobilenet.layers[i].output.shape);
  // }

//console.log(cn1, cn2);


  const input1 = mobilenet.output;
const input2 =  mobilenet.output;
const addLayer = tf.layers.add();
const sum = addLayer.apply([input1, input2]);


const multiplyLayer = tf.layers.multiply();
const product = multiplyLayer.apply([input1, input2]);
const cnet = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
    
   //   tf.layers.flatten({inputShape: [7,7,256]}),
//tf.layers.lstm({units: 5, returnSequences: true}),
//tf.layers.dropout({rate: 0.01}),
//tf.layers.simpleRNN({units: 2, returnSequences: true,inputShape: [1,7,7,256]}),
    // 
    //  tf.layers.flatten({inputShape: [12544,2]}),
    tf.layers.dense({inputShape: [12544,2],
        units: 75,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      
       tf.layers.flatten(),
     tf.layers.dense({
        units: 30,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
     tf.layers.dense({
        units: 10,
        activation: 'softmax',
        kernelInitializer: 'varianceScaling',
        useBias: true
      })
    ]
  });

//console.log(cnet);


//clayer = cnet.getLayer("dense_Dense2");
 

  var countnet=tf.model({inputs: cnet.inputs, outputs: cnet.outputs})

  //await countnet.save('indexeddb://countnet');
  return countnet;
  }
}

// When the UI buttons are pressed, read a frame from the webcam and associate
// it with the class label given by the for example, bananas, oranges are
// labels 0, 1 respectively.
async function setExampleHandler() {
  if(document.querySelector('.testCount').value==''){
    throw 'this sample does not have a count definition';
    return
  }
  // console.log('Adding samples for ',label)
  tf.tidy(() => {

      getObjectStore('data', 'readwrite').get('countNet-dataset').onsuccess = function (event) {
        var reqs = event.target.result;
        try {
            reqs = JSON.parse(reqs);
        } catch (err) {
            reqs = []
        };



      reqs.push({sImg:getCanvasFrame('single').toDataURL(),mImg:getCanvasFrame('many').toDataURL(),count:parseInt(document.querySelector('.testCount').value)})



var setdb = getObjectStore('data', 'readwrite').put(JSON.stringify(reqs), 'countNet-dataset');
            setdb.onsuccess = function () {
                
    //console.log('sample added');
    return 'sample added';
    //modelDataset.addExample(product, parseInt(document.querySelector('input').value));

            }

      }

  });
}


async function addData(){
  getObjectStore('data', 'readwrite').get('countNet-dataset').onsuccess = function (event) {
        var reqs = event.target.result;
        try {
            reqs = JSON.parse(reqs);
        } catch (err) {
            reqs = []
        };


const multiplyLayer = tf.layers.multiply();
document.querySelector('.collection').innerHTML='';
for(var i in reqs){

document.querySelector('.collection').appendChild(htmlToElement('<li style="padding-left: 144px;" class="collection-item avatar">'+
      '<img src="'+reqs[i].sImg+'" alt="" class="materialboxed circle">'+
     '<img style="margin-left: 60px;" src="'+reqs[i].mImg+'" alt="" class="materialboxed">'+
      '<span class="title">'+reqs[i].count+' Items</span>'+
      '<p> <br></p>'+
      '<a href="#!" class="secondary-content"><i class="material-icons">grade</i></a>'+
    '</li>'));
  

  new Promise(function (resolve, reject) {


var singleCanv = document.createElement('canvas');
var singleCtx = singleCanv.getContext("2d");
var manyCanv = document.createElement('canvas');
var manyCtx = manyCanv.getContext("2d");
var cnt=reqs[i].count;
var thNum=i;

  singleCanv.width = 224;
  singleCanv.height =  224;
  manyCanv.width = 224;
  manyCanv.height =  224;
  
      var sImage = new Image();
sImage.onload = function() {
 
singleCtx.drawImage(sImage, 0, 0, 224, 224);



var mImage = new Image();
mImage.onload = function() {
 
manyCtx.drawImage(mImage, 0, 0, 224, 224);

//add this sample to dataset

    var singleImg = new Webcam(singleCanv).capture();
    var manyImg = new Webcam(manyCanv).capture();
    // console.log(img.data())
    
var product = tf.stack([mobilenet.predict(singleImg).reshape([1,12544]), mobilenet.predict(manyImg).reshape([1,12544])], 2);
//var product = mobilenet.predict(singleImg).squaredDifference(mobilenet.predict(manyImg));
resolve({product:product,count:parseInt(cnt),thNum:thNum})
  

};
mImage.src = reqs[i].mImg;


};
sImage.src = reqs[i].sImg;

    }).then(function(e){

modelDataset.addExample(e.product, e.count);
console.log(e.thNum,i)
if(e.thNum==i){

return modelDataset;
}

    })
  

//console.log(manyCtx,reqs[i].mImg)

   // if(i++==reqs.length)return true;

}


      }

await delay()
}
trainingRun=0;
train = async function() {

  ++trainingRun;    

  if (modelDataset.xs == null) {
    throw new Error('Add some examples before training!');
  }
  //sokoCustomModelInit()

   // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(0.001);
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  countnet.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });


  const lossValues = [];
  const accuracyValues = [];

  // We parameterize batch size as a fraction of the entire dataset because the
  // number of examples that are collected depends on how many examples the user
  // collects. This allows us to have a flexible batch size.
  const batchSize = Math.floor(modelDataset.xs.shape[0] * 1);
  if (!(batchSize > 0)) {
    throw new Error(`Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.

  console.log(Math.floor(modelDataset.xs.shape[0] * 1))
//document.querySelector('.modelStats').innerHTML = '';
  for (let i = 0; i < 100; i++) {
    const history = await countnet.fit(modelDataset.xs, modelDataset.ys, {batchSize: batchSize,epochs: 1});
    const loss = history.history.loss[0];
    const accuracy = history.history.acc[0];

    if(i==0){
document.querySelector('.modelStats').prepend(document.createTextNode('trainCount:'+trainingRun+' '+'epoch:'+i+' '+'loss:'+loss+' '+'acc:'+accuracy+'</br>'));
 

    }else if(i==99){
 document.querySelector('.modelStats').prepend(document.createTextNode('trainCount:'+trainingRun+' '+'epoch:'+i+' '+'loss:'+loss+' '+'acc:'+accuracy+'</br>'));
   	
    }

   //console.log({'loss':loss,'acc':accuracy});
    await tf.nextFrame();

  } 
  await countnet.save('indexeddb://countnet');

  return 'trained and saved';     
  //indexeddb://demo/management/model1
}


let Predicting = true;
var generatingTestData=false;
//update();
setInterval(function() {
  
  if(generatingTestData) {
  	
  manySamples(30).then(function(e){
  	train().then(function(e){
  	console.log(e);
  })
  })
  
  }
}, 15000);


async function manySamples(num){
	 for (var i = 0; i < num; ++i) {
            await createTestData(getRandomInt(1, 2),getRandomInt(1, 9));
            await delay();


                        }

}

//manySamples(10)

function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}
async function predict() {
 // while (Predicting) {
    const predictedClass = tf.tidy(() => {
const singleImg = new Webcam(getCanvasFrame('single')).capture();
    const manyImg = new Webcam(getCanvasFrame('many')).capture();
    // console.log(img.data())

const multiplyLayer = tf.layers.multiply();

//var product = tf.concat([mobilenet.predict(singleImg), mobilenet.predict(manyImg)], 0);
var product = mobilenet.predict(singleImg).squaredDifference(mobilenet.predict(manyImg));
//const product = multiplyLayer.apply([mobilenet.predict(singleImg), mobilenet.predict(manyImg)]);
console.log(product)
      
      // Make a prediction through our newly-trained model using the activation
      // from mobilenet as input.
      const predictions = countnet.predict(product);
      //returns boxxes and corresponding classes
      return predictions;
    });

    const activations = (await predictedClass);
    console.log(activations.print());
    console.log(activations.dataSync());
    activations.dispose()
    await tf.nextFrame();
  //}
  
  
}
allowAutoTrain=false;
async function autoTrain(){

setInterval(function() {
  if(allowAutoTrain){
  	try{
  	}catch(e){
  		console.log('countnet model not loaded for retraining')
  	}
allowAutoTrain=false;
idbDatabase.transaction(["data"], "readwrite").objectStore("data").clear()
try{
	// The dataset object where we will store activations.
modelDataset = new ModelDataset();

}catch(e){
	console.log(e)
}
  manySamples(4).then(function(e){
addData().then(function(e){
  	console.log(e);
  	train().then(function(e){
  	console.log(e);

allowAutoTrain=true;
  })
  });
  })
}
}, 5000);

  
}
autoTrain();


async function init() {
countnet =  await loadCountNet();
mobilenet =  await loadMobilenet();
datasetManager();

var elems = document.querySelectorAll('.materialboxed');
    var instances = M.Materialbox.init(elems, {});
/*
  manySamples(10).then(function(e){
addData().then(function(e){
	
  	console.log(e);

var elems = document.querySelectorAll('.materialboxed');
    var instances = M.Materialbox.init(elems, {});

  	train().then(function(e){
  	console.log(e);
  })
  });
  })
  */
}
setTimeout(function() {
  init()
}, 1000);


var tabsInstance = M.Tabs.init(document.querySelector('.tabs'), {});
  

//toggle test data generation  
document.querySelector(".autoTrain>label>input").addEventListener("click", function(e){
if(generatingTestData){
allowAutoTrain=false;
addData()
}else{


allowAutoTrain=true;
}
});