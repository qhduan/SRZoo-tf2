
const tf = require('@tensorflow/tfjs-node')
const fs = require('fs')
const jpeg = require('jpeg-js')


const readImage = async path => {
    const buf = fs.readFileSync(path)
    const pixels = jpeg.decode(buf, true)
    return pixels
}

const imageByteArray = async (image, numChannels) => {
    const pixels = image.data
    const numPixels = image.width * image.height;
    const values = new Int32Array(numPixels * numChannels);

    for (let i = 0; i < numPixels; i++) {
        for (let channel = 0; channel < numChannels; ++channel) {
            values[i * numChannels + channel] = pixels[i * 4 + channel];
        }
    }

    return values
}

const imageToInput = async (image, numChannels) => {
    const values = await imageByteArray(image, numChannels)
    const outShape = [1, image.height, image.width, numChannels]
    const input = tf.tensor4d(values, outShape, 'float32')

    return input
}


(async () => {
    const imagePath = 'input.jpg'
    const modelPath = './outputs/edsr_baseline_x4'



    let pixel_max_one = false
    if (modelPath.match(/carn|dbpn|esrgan|frsr|natsr|rrdb/)) {
        pixel_max_one = true
    }

    let channel_first = true
    if (modelPath.match(/4pp_eusr|eusr|frsr|natsr/)) {
        channel_first = false
    }

    try {
        const image = await readImage(imagePath)
        let input = await imageToInput(image, 3)

        if (channel_first) {
            input = tf.transpose(input, [0, 3, 1, 2])
        }
        if (pixel_max_one) {
            input = input / 255.0
        }

        const model = await tf.node.loadSavedModel(modelPath)
        const out = model.predict(input)

        // console.log(out)

        // let p = await out.arraySync()
        let p = out

        if (channel_first) {
            p = tf.transpose(p, [0, 2, 3, 1])
        }
        if (pixel_max_one) {
            p = p * 255.0
        }

        // console.log(p)

        const height = p.shape[1]
        const width = p.shape[2]
        const bufData = await p.buffer()

        const frameData = new Buffer(width * height * 4)
        for (let i = 0; i < height * width; i++) {
            for (let j = 0; j < 3; j++) {
                let v = bufData.values[i * 3 + j]
                v = Math.round(v)
                v = Math.max(0, v)
                v = Math.min(v, 255)
                frameData[i * 4 + j] = v
            }
            frameData[i * 4 + 3] = 0xff
        }

        const outImage = jpeg.encode({
            data: frameData,
            height,
            width,
        }, 100)
        fs.writeFileSync('out.jpg', outImage.data)
    } catch (e) {
        console.error(e)
    }
})()
