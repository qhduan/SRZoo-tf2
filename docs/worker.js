
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs")

let model = null

const imageByteArray = async (image, numChannels) => {
    const pixels = image.data
    const numPixels = image.width * image.height
    const values = new Int32Array(numPixels * numChannels)

    for (let i = 0; i < numPixels; i++) {
        for (let channel = 0; channel < numChannels; ++channel) {
            values[i * numChannels + channel] = pixels[i * 4 + channel]
        }
    }

    return values
}

const imageToInput = async (image, numChannels) => {
    const values = await imageByteArray(image, numChannels)
    const outShape = [1, image.height, image.width, numChannels]
    const inputTensor = tf.tensor4d(values, outShape, 'float32')

    return inputTensor
}


onmessage = async (e) => {
    console.log('Message received from main script', e.data)
    let input = await imageToInput(e.data, 3)
    const model = await tf.loadGraphModel('carn_x2_js/model.json')
    console.log('input', input)
    // carn setting
    let pixel_max_one = true
    let channel_first = true

    if (channel_first) {
        input = await tf.transpose(input, [0, 3, 1, 2])
    }
    if (pixel_max_one) {
        input = tf.div(input, 255.0)
    }

    let out = await model.executeAsync(input)

    if (channel_first) {
        out = tf.transpose(out, [0, 2, 3, 1])
    }
    if (pixel_max_one) {
        out = tf.mul(out, 255.0)
    }

    const height = out.shape[1]
    const width = out.shape[2]
    const data = out.dataSync()
    const ret = {data, height, width}
    console.log('ret', ret)

    postMessage(ret)
}
