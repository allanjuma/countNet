const CANVAS_WIDTH = 224
const CANVAS_HEIGHT = 224 


//should not be hardcoded!
var NUM_CLASSES =2;
var class_names = {}





function htmlToElement(html) {
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

function copySvgToCavnas(){


}


function getVideoFrame(som){
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
    
const y = tf.tidy(() => tf.oneHot(tf.tensor1d([label]).toInt(), 30));


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
function createTestData(shape,count){
      document.querySelector('.singleCanvas').getContext("2d").clearRect(0, 0, document.querySelector('.singleCanvas').width, document.querySelector('.singleCanvas').height);
        document.querySelector('.manyCanvas').getContext("2d").clearRect(0, 0, document.querySelector('.manyCanvas').width, document.querySelector('.manyCanvas').height);
    
  document.querySelector('.testCount').value=count;
update(shape,count).then(function(e){

    // this is just a JavaScript (HTML) image
    var img = new Image();
    // http://en.wikipedia.org/wiki/SVG#Native_support
    // https://developer.mozilla.org/en/DOM/window.btoa
    
    img.onload = function() {
        // after this, Canvas’ origin-clean is DIRTY
document.querySelector('.manyCanvas').getContext("2d").drawImage(img, 0, 0);
        update(getRandomInt(1, 2),1).then(function(e){

   
    // this is just a JavaScript (HTML) image
    var imgi = new Image();
    // http://en.wikipedia.org/wiki/SVG#Native_support
    // https://developer.mozilla.org/en/DOM/window.btoa
    
    imgi.onload = function() {
        // after this, Canvas’ origin-clean is DIRTY
        document.querySelector('.singleCanvas').getContext("2d").drawImage(imgi, 0, 0);
        setExampleHandler();
    }
    imgi.src = "data:image/svg+xml;base64," + btoa((new XMLSerializer()).serializeToString(document.querySelector('svg')));

  })
    }
    
img.src = "data:image/svg+xml;base64," + btoa((new XMLSerializer()).serializeToString(document.querySelector('svg')));

  })
 

    
}

// The dataset object where we will store activations.
let modelDataset = new ModelDataset();

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

cnet = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten({inputShape: [7,7,256]}),
      // Layer 1
      tf.layers.dense({
        units: 256,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // Layer 2. The number of units of the last layer should correspond
      tf.layers.dense({
        units: 30,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

console.log(cnet);


clayer = cnet.getLayer("dense_Dense2");
 

  return tf.model({inputs: cnet.inputs, outputs: clayer.output});
}

// When the UI buttons are pressed, read a frame from the webcam and associate
// it with the class label given by the for example, bananas, oranges are
// labels 0, 1 respectively.
function setExampleHandler() {
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
                
    console.log('sample added')
    //modelDataset.addExample(product, parseInt(document.querySelector('input').value));

            }

      }

  });
}

function sokoCustomModelInit(){
  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.

model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten({inputShape: [7, 7, 256]}),
      // Layer 1
      tf.layers.dense({
        units: 500,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // Layer 2. The number of units of the last layer should correspond
       
      tf.layers.dense({
        units: 150,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // Layer 3. The number of units of the last layer should correspond
        
      tf.layers.dense({
        units: 50,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // Layer 3. The number of units of the last layer should correspond
      // to the number of classes we want to predict.
      tf.layers.dense({
        units: 8,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });


  // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(0.001);
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
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
     '<img style="margin-left: 60px;" src="'+reqs[i].mImg+'" alt="" class="materialboxed circle">'+
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

var product = multiplyLayer.apply([mobilenet.predict(singleImg), mobilenet.predict(manyImg)]);
resolve({product:product,count:parseInt(cnt)})
  

};
mImage.src = reqs[i].mImg;


};
sImage.src = reqs[i].sImg;

    }).then(function(e){
modelDataset.addExample(e.product, e.count);


    })
  

//console.log(manyCtx,reqs[i].mImg)

   // if(i++==reqs.length)return true;

}


      }

}

async function train() {

      

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
document.querySelector('.modelStats').innerHTML = '';
  for (let i = 0; i < 100; i++) {
    const history = await countnet.fit(modelDataset.xs, modelDataset.ys, {batchSize: batchSize,epochs: 1});
    const loss = history.history.loss[0];
    const accuracy = history.history.acc[0];

document.querySelector('.modelStats').appendChild(document.createTextNode('loss:'+loss+' '+'acc:'+accuracy+'\n'));
    //console.log({'loss':loss,'acc':accuracy});
    await tf.nextFrame();

  }      
}


let Predicting = true;
var generatingTestData=false;
//update();
setInterval(function() {
  
  if(generatingTestData) {
   createTestData('',getRandomInt(1, 30))
  }
}, 2000);


function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}
async function predict() {
  while (Predicting) {
    const predictedClass = tf.tidy(() => {
const singleImg = new Webcam(getCanvasFrame('single')).capture();
    const manyImg = new Webcam(getCanvasFrame('many')).capture();
    // console.log(img.data())

const multiplyLayer = tf.layers.multiply();
const product = multiplyLayer.apply([mobilenet.predict(singleImg), mobilenet.predict(manyImg)]);
console.log(product)
      
      // Make a prediction through our newly-trained model using the activation
      // from mobilenet as input.
      const predictions = countnet.predict(product);
      //returns boxxes and corresponding classes
      return predictions;
    });

    const activations = (await predictedClass);
    console.log(activations.dataSync());
    activations.dispose()
    await tf.nextFrame();
  }
  
  
}


async function init() {
countnet =  await loadCountNet();
mobilenet =  await loadMobilenet();
addData().then(function(r){

setInterval(function() {
  

var elems = document.querySelectorAll('.materialboxed');
    var instances = M.Materialbox.init(elems, {});
}, 2000);
   // startWebcam(document.querySelector('video'));
})
}
init()

var tabsInstance = M.Tabs.init(document.querySelector('.tabs'), {});
  

//toggle test data generation  
document.querySelector(".testDataGen>label>input").addEventListener("click", function(e){
if(generatingTestData){
generatingTestData=false;
addData()
}else{


generatingTestData=true;
}
});