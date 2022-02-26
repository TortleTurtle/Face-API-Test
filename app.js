videoInput = document.getElementById("videoInput");

window.addEventListener('load', () =>{
	Promise.all([
		faceapi.nets.tinyFaceDetector.loadFromUri("./models"),
		faceapi.nets.faceRecognitionNet.loadFromUri("./models"),
		faceapi.nets.faceLandmark68Net.loadFromUri("./models"),
		faceapi.nets.ssdMobilenetv1.loadFromUri("./models")
	]).then(start);
});

async function start () {
	console.log("Loaded models");

	//load labeled images for face recognition
	console.log('loading reference images');
	const labeledFaceDescriptors = await loadLabeledImages();
	const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6); //60%

	await startVideo();

	videoInput.addEventListener('play', () => {
		//create a canvas to display detections
		const canvas = faceapi.createCanvasFromMedia(videoInput);
		document.body.append(canvas);
		const displaySize = {width: videoInput.width, height: videoInput.height }
		faceapi.matchDimensions(canvas, displaySize);

		//set an interval to check for detections
		setInterval( async () => {
			//detect all faces. Using TinyFace because its more light weight.
			const detections = await faceapi.detectAllFaces(videoInput, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptors();
			const resizedDetections = faceapi.resizeResults(detections, displaySize);
			//find the best match for the detected faces.
			const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));

			//clear canvas to display new detections.
			canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
			//draw results
			results.forEach((result, i) => {
				const box = resizedDetections[i].detection.box;
				const drawBox = new faceapi.draw.DrawBox(box, {label: result.toString()});
				drawBox.draw(canvas);
			});
		}, 100) // 1/10th of a second.
	});
}

function loadLabeledImages() {
	//labels for images. These need to correspond with the folder names where the training images are stored.
	const labels = ['Benedict_Cumberbatch', 'Chris_Hemsworth', 'Jennifer_Lawrence', 'RDJ', 'Sigourney_Weaver'];

	return Promise.all(
		//map labels and match with corresponding descriptors.
		labels.map(async label => {
			const descriptions = [];

			for (let i = 1; i <= 6; i++) {
				//using npm live-server to host files locally.
				const img = await faceapi.fetchImage(`http://127.0.0.1:8080/labeled_images//${label}/${i}.jpg`);
				const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
				console.log(detections);
				descriptions.push(detections.descriptor);
			}

			return new faceapi.LabeledFaceDescriptors(label, descriptions);
		})
	);
}

async function startVideo() {
	//get access to camera and link it to video element.
	const constraints = {
		audio: false,
		video: {
			width: videoInput.width,
			height: videoInput.height
		}
	}

	try {
		stream = await navigator.mediaDevices.getUserMedia(constraints);
		videoInput.srcObject = stream;
	} catch (err) {
		console.error(err);
	}
}