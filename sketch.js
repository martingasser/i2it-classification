let model
let xSel
let ySel
let modelTrained = false

function setupUI(data) {

    let canvas = createCanvas(400, 400)

    let trainButton = createButton('Train model')
    trainButton.position(10, 10)
    trainButton.mousePressed(() => {
        trainModel(model, data).then(() => {
            console.log('Training finished')
            modelTrained = true
            background(0)
            const input = createFileInput(handleFile);
            input.position(100, 10);
        })
    });

    canvas.position(10, 100)
}

function handleFile(file) {
    if (file.type === 'image') {
        let urlOfImageFile = URL.createObjectURL(file.file)
        let imageObject = loadImage(urlOfImageFile, () => {
            
            imageObject.loadPixels()

            const datasetBytesBuffer = new ArrayBuffer(IMAGE_SIZE * 4);
            const datasetBytesView = new Float32Array(datasetBytesBuffer, 0, IMAGE_SIZE);

            for (let j = 0; j < imageObject.pixels.length / 4; j++) {
                datasetBytesView[j] = imageObject.pixels[j * 4] / 255;
            }

            const xs = tf.tensor2d(datasetBytesView, [1, IMAGE_SIZE])
            const prediction = predict(model, xs)        

            image(imageObject, 0, 0, width, height)
        })
    }
}


function setup() {
    model = createTFModel()
    const data = new MnistData()
    data.load().then(() => {
        setupUI(data)
    })
}

function draw() {
}
