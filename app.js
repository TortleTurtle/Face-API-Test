imageUpload = document.getElementById("imageUpload");

window.addEventListener('load', () =>{
	Promise.all([
		faceapi.nets.faceRecognitionNet.loadFromUri("./models"),
		faceapi.nets.faceLandmark68Net.loadFromUri("./models"),
		faceapi.nets.ssdMobilenetv1.loadFromUri("./models"),
	]).then(start);
});

async function start () {
	console.log("Loaded!");

	//create containter to hold image and canvas
	const container = document.createElement('div');
	container.style.position = "relative";
	document.body.append(container);

	//variables to store image and canvas
	let image
	let canvas

	//load labeled images for face recognition
	const labeledFaceDescriptors = await loadLabeledImages();
	const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6); //60%

	imageUpload.addEventListener('change', async () => {
		//clear image and canvas if already exists to clean up the page.
		if (image) image.remove();
		if (canvas) canvas.remove();

		//buffer uploaded image and append to container.
		console.log("Buffering Image")
		image = await faceapi.bufferToImage(imageUpload.files[0]);
		console.log("Done!");
		container.append(image);

		//create canvas, resize to image and append to container.
		canvas = faceapi.createCanvasFromMedia(image);
		container.append(canvas);
		const displaySize = { width: image.width, height: image.height };
		faceapi.matchDimensions(canvas, displaySize);

		//get detections
		const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors();
		const resizedDetections = faceapi.resizeResults(detections, displaySize);
		const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))

		//draw detections
		results.forEach((result, i) => {
			const box = resizedDetections[i].detection.box;
			const drawBox = new faceapi.draw.DrawBox(box, {label: result.toString()});
			drawBox.draw(canvas);
		});
		console.log(detections.length);
	});
}

function loadLabeledImages() {
	//labels for images
	const labels = ['Coen', 'Daan'];
	// const labels = ['Black Widow', 'Captain America', 'Captain Marvel', 'Hawkeye', 'Jim Rhodes', 'Thor', 'Tony Stark'];

	return Promise.all(
		//map labels and match with corresponding descriptors.
		labels.map(async label => {
			const descriptions = [];

			//Huisgenoten
			for (let i = 1; i <= 4; i++) {
				const img = await faceapi.fetchImage(`http://127.0.0.1:8080/labeled_images//${label}/${i}.jpg`);
				const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
				console.log(detections);
				descriptions.push(detections.descriptor);
			}

			//Avengers
			// for (let i = 1; i <= 2; i++){
			// 	const img = await faceapi.fetchImage(`https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/${label}/${i}.jpg`);
			// 	const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
			// 	console.log(detections);
			// 	descriptions.push(detections.descriptor);
			// }
			return new faceapi.LabeledFaceDescriptors(label, descriptions);
		})
	)
}