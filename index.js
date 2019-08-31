const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');

let net;

// webcam setup function for inference in browser through webcam images
async function setupWebcam() {
    return new Promise((resolve, reject) => {
        const navigatorAny = navigator;
        navigator.getUserMedia = (navigator.getUserMedia ||
            navigatorAny.webkitGetUserMedia || 
            navigatorAny.mozGetUserMedia ||
            navigatorAny.msGetUserMedia);
            
        if (navigator.mediaDevices) { // if navigator.mediaDevices exists, use it
            // adapted code compatible with Safari 12
            // code could be optimized by adding variables onGetUserMedia and onGetUserMediaError
            navigator.mediaDevices.getUserMedia({video: true}).then(
                stream => {
                    webcamElement.srcObject = stream;
                    webcamElement.addEventListener('loadeddata', () => resolve(), false);
                }, 
                error => reject());
        } else {
            reject();
        }
    });
} 

// set up MobileNet for inference in browser
async function app() {
    console.log('Loading mobilenet..');

    // Load the model.
    net = await mobilenet.load();
    console.log('Successfully loaded model');

    /*
    // Make a prediction through the mode on our image.
    const imgEl = document.getElementById('img');
    const result = await net.classify(imgEl);
    console.log(result);
    */

    await setupWebcam();
    
    /*
    // Set up for the webcamElement (without the KNN classifier)
    while (true) {
        const result = await net.classify(webcamElement);

        document.getElementById('console').innerText = `
            prediction: ${result[0].className}\n
            probability: ${result[0].probability}
        `;

        // Give some breathing room by waiting for the next animation frame to
        // fire.
        await tf.nextFrame();
    }
    */

    // Set up for the webcamElement with the KNN classifier
    // Reads an image from the webcam and associates it with a specific class
    // index.
    const addExample = classId => {
        // Get the intermediate activation of MobileNet 'conv_preds' and pass that
        // to the KNN classifier.
        const activation = net.infer(webcamElement, 'conv_preds');

        // Pass the intermediate activation to the classifier.
        classifier.addExample(activation, classId);
    };

    // When clicking a button, add an example for that class.
    document.getElementById('class-a').addEventListener('click', () => addExample(0));
    document.getElementById('class-b').addEventListener('click', () => addExample(1));
    document.getElementById('class-c').addEventListener('click', () => addExample(2));
    document.getElementById('class-d').addEventListener('click', () => addExample(3));

    while (true) {
        if (classifier.getNumClasses() > 0) {
            // Get the activation from mobilenet from the webcam.
            const activation = net.infer(webcamElement, 'conv_preds');
            // Get the most likely class and confidences from the classifier module.
            const result = await classifier.predictClass(activation);

            const classes = ['A', 'B', 'C', 'D'];
            document.getElementById('console').innerText = `
                prediction: ${classes[result.classIndex]}\n
                probability: ${result.confidences[result.classIndex]}
            `;
        }

        await tf.nextFrame();
    }
}

app();